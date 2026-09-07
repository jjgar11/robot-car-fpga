class MovementPipe:
    """
    Future movement subsystem.

    Replace this stub with the ARM-side motor/sensor implementation.
    """

    def start(self):
        pass

    def on_prediction(
        self,
        snapshot,
    ):
        pass

    def stop(self):
        pass


class CommunicationPipe:
    """
    Future communication subsystem.

    Replace publish() with the selected communication protocol later.
    """

    def start(self):
        pass

    def publish(
        self,
        snapshot,
    ):
        pass

    def stop(self):
        pass
