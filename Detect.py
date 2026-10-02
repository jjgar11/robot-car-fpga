import torch
from torch import nn
from torch.nn.functional import interpolate
from torch.nn import Conv2d, Linear, ReLU, Sequential, ModuleDict, Sigmoid, MaxPool2d
from torchvision.transforms.v2.functional import pad, to_pil_image
from torchvision.transforms.v2 import Resize


MAX_PATCH_SIZE = 256    # also used as initial patch size
MIN_PATCH_SIZE = 64 # iteration stops after min size is reached
PATCH_SCALE = 0.5   # size reduction at each iteration
PATCH_STRIDE = 1  # stride relative to patch size
MAX_ITERATIONS = 3  # stops iterating if min size isn't reached before
MIN_SIGN_SIZE = 100 # filter out far away signs


resize = Resize(64)


class Classifier(nn.Module):
    def __init__(self, filename=None):
        super().__init__()
        # setup model
        self.optimizer = None
        self.loss_fn = nn.CrossEntropyLoss()
        features = nn.Sequential(Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
                                 ReLU(),
                                 Conv2d(16, 16, kernel_size=2, stride=2, bias=False, groups=16),
                                 ReLU(),
                                 Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
                                 Conv2d(32, 32, kernel_size=2, stride=2, bias=False, groups=32),
                                 ReLU(),
                                 Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                                 ReLU(),
                                 Conv2d(64, 64, kernel_size=2, stride=2, bias=False, groups=64),
                                 ReLU(),
                                 Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                                 ReLU(),
                                 Conv2d(128, 128, kernel_size=4, stride=4, bias=False, groups=128),
                                 ReLU())
        classifier = nn.Sequential(Linear(512, 64, bias=False),
                                   ReLU(),
                                   Linear(64, 44, bias=False))
        self.model = nn.ModuleDict({"features": features, "classifier": classifier})
        # load model
        if filename is not None:
            self.load(filename)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.flatten(x, 1)
        return self.model.classifier(x)

    def get_loss(self, x, target):
        return self.loss_fn(self(x), target)

    def training_step(self, batch, batch_idx):
        inpt, target = batch
        loss = self.loss_fn(self(inpt), target)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, target = batch
        assert x.size(0) == 1, "testing must be done with batch size 1"
        logits = self(x)
        result = torch.argmax(logits, dim=1)
        target_is_sign = torch.gt(target, 0)
        result_is_sign = torch.gt(result, 0)
        is_correct = torch.eq(result, target)
        if torch.sum(target_is_sign) > 0:
            recall = (target_is_sign * result_is_sign) / torch.sum(target_is_sign)
            class_accuracy = (target_is_sign * is_correct) / torch.sum(target_is_sign)
            self.log("detection_recall", recall)
            self.log("classification_accuracy", class_accuracy)
        if torch.sum(result_is_sign) > 0:
            precision = (target_is_sign * result_is_sign) / torch.sum(result_is_sign)
            self.log("detection_precision", precision)
        corr = torch.sum(is_correct).item()
        self.log("accuracy", corr / target.size(0))

    def configure_optimizers(self):
        if self.optimizer is None:
            return torch.optim.Adam(self.model.parameters())
        else:
            return self.optimizer

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict)


class Locator(nn.Module):
    def __init__(self, filename=None, threshold_exclusion=0.5, threshold_detection=0.5):
        super().__init__()
        self.threshold_exclusion = torch.tensor(threshold_exclusion)
        self.threshold_detection = torch.tensor(threshold_detection)
        # setup model
        features = Sequential(Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
                              ReLU(),
                              MaxPool2d(kernel_size=2, stride=2),
                              Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
                              ReLU(),
                              MaxPool2d(kernel_size=2, stride=2),
                              Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                              ReLU(),
                              MaxPool2d(kernel_size=2, stride=2),
                              Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                              ReLU(),
                              MaxPool2d(kernel_size=2, stride=2))
        self.optimizer = None
        classifier = Sequential(Linear(2048, 512, bias=False),
                                ReLU(),
                                Linear(512, 5, bias=True),
                                Sigmoid())
        self.model = ModuleDict({"features": features, "classifier": classifier})
        # load model
        if filename is not None:
            self.load(filename)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.flatten(x, 1)
        return self.model.classifier(x)

    def training_step(self, batch, batch_idx):
        x, target = batch
        result = self(x)
        loss_exclusion = torch.mean(torch.pow(result[:, 0] - target[:, 0], 2))
        loss_detection = torch.mean(torch.pow(result[:, 1] - target[:, 1], 2))
        dist = torch.sqrt(torch.pow(result[:, 2] - target[:, 2], 2) + torch.pow(result[:, 3] - target[:, 3], 2) + 1e-5)
        loss_position = torch.mean(dist * target[:, 1])
        loss_scale = torch.mean(torch.pow((target[:, 4] - result[:, 4]), 2) * target[:, 1])
        loss = loss_exclusion + loss_detection + loss_position + loss_scale
        return loss

    def test_step(self, batch, batch_idx):
        x, target = batch
        result = self(x)
        pred_exclusion = torch.ge(result[:, 0], self.threshold_exclusion)
        target_exclusion = target[:, 0]
        tp_exclusion = pred_exclusion * target_exclusion
        if torch.sum(pred_exclusion) > 0:
            precision_exclusion = torch.sum(tp_exclusion) / torch.sum(pred_exclusion)
            self.log("precision_exclusion", precision_exclusion)
        if torch.sum(target_exclusion) > 0:
            recall_exclusion = torch.sum(tp_exclusion) / torch.sum(target_exclusion)
            self.log("recall_exclusion", recall_exclusion)
        pred_detection = torch.ge(result[:, 1], self.threshold_detection)
        target_detection = target[:, 1]
        tp_detection = pred_detection * target_detection
        if torch.sum(pred_detection) > 0:
            precision_detection = torch.sum(tp_detection) / torch.sum(pred_detection)
            self.log("precision_detection", precision_detection)
        if torch.sum(target_detection) > 0:
            recall_detection = torch.sum(tp_detection) / torch.sum(target_detection)
            self.log("recall_detection", recall_detection)
        pred_crop = result[:, 2:]
        target_crop = target[:, 2:]
        pos_diff = torch.sqrt(torch.pow(pred_crop[:, 0] - target_crop[:, 0], 2) + torch.pow(pred_crop[:, 1] - target_crop[:, 1], 2)) * (target[:, 1] * pred_detection)
        size_diff = torch.abs(pred_crop[:, 2] - target_crop[:, 2]) * (target[:, 1] * pred_detection)
        for p_d in pos_diff:
            if p_d > 0:
                self.log("pos_diff", torch.mean(p_d))
        for s_d in size_diff:
            if s_d > 0:
                self.log("size_diff", torch.mean(s_d))

    def configure_optimizers(self):
        if self.optimizer is None:
            return torch.optim.Adam(self.model.parameters())
        else:
            return self.optimizer

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict)


def split_patch(patch_x, patch_y, patch_size):
    sub_patches = []
    next_patch_size = patch_size * PATCH_SCALE
    num_pos = int((patch_size / next_patch_size)) * (int(1 / PATCH_STRIDE)) - (int(1 / PATCH_STRIDE) - 1)
    for i_y in range(num_pos):
        for i_x in range(num_pos):
            next_patch_x = patch_x + (i_x * next_patch_size * PATCH_STRIDE)
            next_patch_y = patch_y + (i_y * next_patch_size * PATCH_STRIDE)
            sub_patches.append((int(next_patch_x), int(next_patch_y), int(next_patch_size)))
    return sub_patches


@torch.no_grad()
def run_inference(image, locator: Locator, label_func):
    detected_signs = []
    # enforce batch size of 1
    assert image.size(0) == 1, "batch size of 1 required during inference"
    # prepare image
    image_width = torch.tensor(image.size(-1))
    image_height = torch.tensor(image.size(-2))
    # pad image to full multiple of patch size
    pad_top = int((torch.ceil(image_height / MAX_PATCH_SIZE) * MAX_PATCH_SIZE) - image_height)
    pad_left = int((torch.ceil(image_width / MAX_PATCH_SIZE) * MAX_PATCH_SIZE) - image_width)
    padded_image = pad(image, [pad_left, pad_top, 0, 0])
    # generate initial set of patches
    next_patches = []   # x, y, size
    num_x_pos = int(((padded_image.size(-1) / MAX_PATCH_SIZE) * (int(1 / PATCH_STRIDE))) - (int(1 / PATCH_STRIDE) - 1))
    num_y_pos = int(((padded_image.size(-2) / MAX_PATCH_SIZE) * (int(1 / PATCH_STRIDE))) - (int(1 / PATCH_STRIDE) - 1))
    abs_patch_stride = MAX_PATCH_SIZE * PATCH_STRIDE
    for pos_y in range(num_y_pos):
        for pos_x in range(num_x_pos):
            patch_x = int(pos_x * abs_patch_stride)
            patch_y = int(pos_y * abs_patch_stride)
            next_patches.append((patch_x, patch_y, MAX_PATCH_SIZE))
    current_iteration = 1
    # iterate through patches
    while len(next_patches) > 0:
        # check for maximum number of iterations
        if current_iteration >= MAX_ITERATIONS:
            break
        current_patches = [patch for patch in next_patches]
        next_patches = []
        # crop and resize all square patches of this level at once (bit-exact with per-patch Resize(64))
        resized_patches = interpolate(torch.cat([padded_image[:, :, y:y + s, x:x + s] for x, y, s in current_patches]),
                                      size=[64, 64], mode="bilinear", align_corners=False, antialias=True)
        for (patch_x, patch_y, patch_size), resized_patch in zip(current_patches, resized_patches):
            # apply locator model
            result = torch.squeeze(locator(resized_patch[None]))
            # unpack results
            exclusion_score = result[0]
            detection_score = result[1]
            sign_x = result[2]
            sign_y = result[3]
            sign_size = result[4]
            # check if patch has sign candidate
            if detection_score > locator.threshold_detection:
                # crop sign candidate
                abs_sign_x = ((sign_x * patch_size) + patch_x).detach().clone().to(dtype=torch.int16)
                abs_sign_y = ((sign_y * patch_size) + patch_y).detach().clone().to(dtype=torch.int16)
                abs_sign_size = ((sign_size * patch_size)).detach().clone().to(dtype=torch.int16)
                # get sign top left corner coordinates, clamp to prevent getting out of bounds
                sign_corner_x = torch.clamp(torch.round(abs_sign_x - (0.5 * abs_sign_size)).to(dtype=torch.int16),
                                            torch.tensor(0), padded_image.size(-1) - abs_sign_size)
                sign_corner_y = torch.clamp(torch.round(abs_sign_y - (0.5 * abs_sign_size)).to(dtype=torch.int16),
                                            torch.tensor(0), padded_image.size(-2) - abs_sign_size)
                sign_candidate = padded_image[:, :, sign_corner_y:sign_corner_y + abs_sign_size,
                                             sign_corner_x:sign_corner_x + abs_sign_size]
                # resize and apply classifier model
                if abs_sign_size > 16:
                    scaled_sign_candidate = resize(sign_candidate)
                    infered = label_func(scaled_sign_candidate)
                    label = infered["class_id"]
                else:
                    label = 0
                # if classification successful, add to detections results
                if label > 0:
                    infered["bbox"] = (abs_sign_x, abs_sign_y, abs_sign_size)
                    detected_signs.append(infered)
                # otherwise, split patch and submit for next iteration
                else:
                    next_patches.extend(split_patch(patch_x, patch_y, patch_size))
            # if patch should not be excluded, split and submit for next iteration
            elif exclusion_score < locator.threshold_exclusion:
                next_patches.extend(split_patch(patch_x, patch_y, patch_size))
        current_iteration += 1
    return detected_signs


"""
use like this:
- first, load locator: locator = Locator("path/to/locator.pth")
- grab image from camera
- label_func is a function that returns logits from classification on FPGA; if it returns labels directly as numbers,
replace line 256 by this: label = label_func(scaled_sign_candidate)
- detect() returns list of labels of the detected signs; far away signs are filtered out by MIN_SIGN_SIZE, which
may need some fine-tuning
"""
def detect(image, locator: Locator, label_func):
    sign_candidates = run_inference(image, locator, label_func)
    signs = []
    for sign_candidate in sign_candidates:
        if sign_candidate["bbox"][2] >= MIN_SIGN_SIZE:
            signs.append(sign_candidate)
    return signs