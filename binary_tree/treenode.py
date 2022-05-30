from typing import List


class TreeNode:
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None

    def __str__(self):
        left = None if self.left is None else self.left.val
        right = None if self.right is None else self.right.val
        return '(D:{}, L:{}, R:{})'.format(self.val, left, right)




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


def print2D(root: TreeNode):
    """
    將binary tree印出來
    """
    current_level = [root]
    while current_level:
        print(' '.join(str(node) for node in current_level))
        next_level = list()
        for n in current_level:
            if n.left:
                next_level.append(n.left)
            if n.right:
                next_level.append(n.right)
        current_level = next_level
