def flatten(list_of_lists):
    """
    Flatten a list of lists.

    Args:
        list_of_lists: (list): write your description
    """
    return [item for sublist in list_of_lists for item in sublist]
