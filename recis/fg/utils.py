import json


def dict_lower_case(input_dict, lower_case):
    """Convert dictionary keys to lowercase if specified.

    This utility function conditionally converts all keys in a dictionary
    to lowercase by serializing to JSON, converting the string to lowercase,
    and deserializing back to a dictionary. This is useful for normalizing
    configuration keys to ensure case-insensitive processing.

    Args:
        input_dict (dict): The input dictionary to process.
        lower_case (bool): Whether to convert keys to lowercase.
            If False, returns the input dictionary unchanged.

    Returns:
        dict: The processed dictionary with lowercase keys if lower_case
            is True, otherwise the original dictionary.
    """
    if lower_case:
        dict_str = json.dumps(input_dict).lower()
        input_dict = json.loads(dict_str)
    return input_dict


def parse_multihash(mh_conf):
    """Parse multi-hash configuration string into components.

    This function parses a multi-hash configuration string that defines
    parameters for multi-hash embedding strategies. The configuration
    string follows the format: "prefix:bucket1,bucket2,...:combiner:num".

    Args:
        mh_conf (str): Multi-hash configuration string in the format
            "prefix:bucket1,bucket2,bucket3,bucket4:combiner:num".

    Returns:
        tuple: A tuple containing:
            - prefix (str): Prefix for multi-hash naming
            - buckets (list[int]): List of bucket sizes as integers
            - combiner (str): Combiner strategy (e.g., "concat", "mean")
            - num (int): Number of hash functions

    Raises:
        AssertionError: If the configuration string doesn't have exactly
            4 components separated by colons.
    """
    confs = mh_conf.split(":")
    mh_type = confs[0]
    buckets = [int(x) for x in confs[1].split(",")]
    combiner = confs[2]
    shard = int(confs[3]) == 1
    return mh_type, buckets, combiner, shard


def get_multihash_name(ori_name, prefix, index):
    """Generate multi-hash feature name with prefix and index.

    This function creates a standardized naming convention for multi-hash
    features by combining the original name, prefix, and index.

    Args:
        ori_name (str): Original feature name.
        prefix (str): Multi-hash prefix identifier.
        index (int): Index of the hash function (0-based).

    Returns:
        str: Generated multi-hash feature name in the format
            "{ori_name}_{prefix}_{index}".
    """
    return f"{ori_name}_{prefix}_{index}"


def get_multihash_shared_name(ori_name, prefix, index):
    """Generate multi-hash shared embedding name with prefix and index.

    This function creates a standardized naming convention for shared
    multi-hash embeddings by combining the original name, prefix, and
    1-based index. This is typically used for embedding table sharing
    across different features.

    Args:
        ori_name (str): Original shared embedding name.
        prefix (str): Multi-hash prefix identifier.
        index (int): Index of the hash function (0-based, will be converted to 1-based).

    Returns:
        str: Generated multi-hash shared name in the format
            "{ori_name}-{prefix}{index+1}".
    """
    return f"{ori_name}-{prefix}{index + 1}"
