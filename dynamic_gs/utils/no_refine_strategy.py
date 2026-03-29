class NoRefineStrategy:
    """No-op replacement for gsplat refinement strategies."""

    absgrad = False

    def initialize_state(self, **kwargs):
        del kwargs
        return {}

    def step_pre_backward(self, *args, **kwargs):
        del args, kwargs
        return None

    def step_post_backward(self, *args, **kwargs):
        del args, kwargs
        return None
