

def random_mentor(state, kwargs=None):
    """Take a random action."""
    if kwargs is None:
        kwargs = {}
    num_actions = kwargs.get("num_actions", 4)


def random_safe_mentor(state, kwargs=None):
    """Take a random action, but mask any cliff-stepping actions."""
    if kwargs is None:
        kwargs = {}
    num_actions = kwargs.get("num_actions", 4)


def prudent_mentor(state, kwargs=None):
    """Step away from the closest edge."""
    if kwargs is None:
        kwargs = {}
    num_actions = kwargs.get("num_actions", 4)
