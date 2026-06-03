class Hook:
    """Base class for all hooks in the RecIS training system.

    Hooks provide a way to extend the training process by defining callback methods
    that are called at specific points during training, evaluation, and execution.
    All custom hooks should inherit from this base class and override the relevant
    callback methods.

    The hook system supports the following callback points:
    - Training lifecycle: before_train, after_train
    - Evaluation lifecycle: before_evaluate, after_evaluate
    - Epoch lifecycle: before_epoch, after_epoch
    - Step lifecycle: before_step, after_step
    - Cleanup: end
    """

    def window_mode(self, *args, **kwargs):
        """Called when use window io mode.

        Change arguments for window io run mode.
        """

    def out_off_data(self, *args, **kwargs):
        """Called when out off data iterator."""

    def before_epoch(self, is_train=True, *args, **kwargs):
        """Called before each training epoch starts.

        This method is invoked at the beginning of each training epoch,
        before any steps in that epoch are executed.
        """

    def after_epoch(self, is_train=True, *args, **kwargs):
        """Called after each training epoch completes.

        This method is invoked at the end of each training epoch,
        after all steps in that epoch have been executed.
        """

    def before_window(self, is_train=True, *args, **kwargs):
        """Called before each window."""

    def after_window(self, is_train=True, *args, **kwargs):
        """Called after each window."""

    def before_step(self, is_train=True, *args, **kwargs):
        """Called before each training step.

        This method is invoked before each individual training step
        is executed. Use this for per-step setup logic.
        """

    def after_step(self, is_train=True, *args, **kwargs):
        """Called after each training step completes.

        This method is invoked after each individual training step
        has been executed. Use this for per-step processing logic,
        such as logging metrics or updating statistics.
        """

    def start(self, is_train=True, *args, **kwargs):
        """Called at the very start of the training process."""

    def end(self, is_train=True, *args, **kwargs):
        """Called at the very end of the training process.

        This method is invoked for final cleanup operations,
        such as closing files, finalizing logs, or releasing resources.
        It is called after all other hook methods have completed.
        """

    def after_data(self, is_train=True, *args, **kwargs):
        """Called after each data batch."""

    def after_eval(self, *args, **kwargs):
        """Called after each evaluation run loop."""

    def after_train(self, *args, **kwargs):
        """Called after each training run loop."""

    def after_backward(self, *args, **kwargs):
        """Called after each backward pass."""
