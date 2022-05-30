from typing import List


class TreeNode:
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None


def preorderTraversal(root: TreeNode):
    """
    前序order
    """
    result = []

    def traversal(root: TreeNode):
        if root is None: return
        result.append(root.val)
        traversal(root.left)
        traversal(root.right)

    traversal(root)
    return result


def inorderTraversal(root: TreeNode):
    result = []

    def traversal(root: TreeNode):
        if root is None: return
        traversal(root.left)
        result.append(root.val)
        traversal(root.right)

    traversal(root)
    return result


def postorderTraversal(root: TreeNode):
    result = []

    def traversal(root: TreeNode):
        if root is None: return
        traversal(root.left)
        traversal(root.right)
        result.append(root.val)

    traversal(root)
    return result


def arr2tree(nums: List[int]):
    def createNode(rootIdx: int):
        root = TreeNode(nums[rootIdx])
        if rootIdx * 2 + 1 <= len(nums) - 1 and nums[rootIdx * 2 + 1] is not None:
            root.left = createNode(rootIdx * 2 + 1)

        if rootIdx * 2 + 2 <= len(nums) - 1 and nums[rootIdx * 2 + 2] is not None:
            root.right = createNode(rootIdx * 2 + 2)

        return root

    root = createNode(0)
    return root
