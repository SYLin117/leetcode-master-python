from treenode import *


def levelOrder(root: TreeNode) -> list:
    """
    102.二叉树的层序遍历
    给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）。
    """
    queue = []
    if root is not None: queue.append(root)
    result = []
    while len(queue) != 0:
        size = len(queue)  # 上一level的node數
        vec = []
        for i in range(size):
            node = queue.pop(0)
            if node.val is not None: vec.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(vec)
    return result


def invertTree(root: TreeNode):
    """
    226.翻转二叉树
    """

    def reverseChild(node: TreeNode):
        if node is None: return
        tmp = node.left
        node.left = node.right
        node.right = tmp
        if node.left is not None: reverseChild(node.left)
        if node.right is not None: reverseChild(node.right)

    reverseChild(root)
    return root


def isSameTree(p: TreeNode, q: TreeNode) -> bool:
    """
    100. Same Tree
    """

    def compare(p, q):
        if p is None and q is None: return True
        if p is None and q is not None or p is not None and q is None: return False
        if p.val != q.val: return False
        return compare(p.left, q.left) and compare(p.right, q.right)

    return compare(p, q)


def isSymmetric(root: TreeNode):
    """
    101. 对称二叉树
    解法：遞迴
    """

    def compare(leftnode: TreeNode, rightnode: TreeNode):
        """
        比較兩個node是否相同value
        """
        if leftnode is None and rightnode is not None:
            return False  # 左空右不空
        elif leftnode is not None and rightnode is None:
            return False  # 左不空右空
        elif leftnode is None and rightnode is None:
            return True  # 左空右空
        elif leftnode.val != rightnode.val:
            return False
        else:
            # 此時左右node value相同
            outside = compare(leftnode.left, rightnode.right)  # 比較兩node外側
            inside = compare(leftnode.right, rightnode.left)  # 比較兩node內側
            isSame = outside and inside  # 兩者同時滿足
            return isSame

    if root is None: return True
    return compare(root.left, root.right)


def binaryTreePaths(root: TreeNode) -> list:
    """
    257. 二叉树的所有路径
    给定一个二叉树，返回所有从根节点到叶子节点的路径。
    说明: 叶子节点是指没有子节点的节点。

    解法：递归法+隐形回溯
    """

    def traversal(cur: TreeNode, path: str, result: list[str]) -> None:
        path += str(cur.val)
        # if current node is leave
        if not cur.left and not cur.right:
            return result.append(path)

        if cur.left:  # 隱藏回朔
            traversal(cur.left, path + '->', result)
        if cur.right:  # 隱藏回朔
            traversal(cur.right, path + '->', result)

    path = ''
    result = []
    if not root: return result  # root 為空
    traversal(root, path, result)
    return result


def sumOfLeftLeaves(root: TreeNode) -> int:
    """
    404.左叶子之和
    分辨左leaf：如果左节点不为空，且左节点没有左右孩子，那么这个节点的左节点就是左叶子
    """
    if not root:
        return 0
    leftValue = sumOfLeftLeaves(root.left) if root.left else 0
    rightValue = sumOfLeftLeaves(root.right) if root.right else 0

    midValue = 0
    if root.left is not None and root.left.left is None and root.left.right is None:  # 存在左節點存在且左節點沒有子節點
        midValue = root.left.val
    sum = leftValue + rightValue + midValue
    return sum


def maxDepth(root: TreeNode):
    """
    104.二叉树的最大深度
    给定一个二叉树，找出其最大深度。
    二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
    """
    if root is None:
        return 0
    leftDepth = maxDepth(root.left)
    rightDepth = maxDepth(root.right)
    return 1 + max(leftDepth, rightDepth)


def minDepth(root: TreeNode):
    """
    111.二叉树的最小深度
    最小深度是从根节点到最近叶子节点的最短路径上的节点数量。，注意是叶子节点。

    注意：不可與maxDepth相同
    """
    if root is None: return 0
    leftDepth = minDepth(root.left)
    rightDepth = minDepth(root.right)

    if (root.left is None and root.right is not None):  # 只有右子樹
        return 1 + rightDepth

    if (root.right is None and root.left is not None):  # 只有左子樹
        return 1 + leftDepth

    result = 1 + min(leftDepth, rightDepth)
    return result


def getNodeNum1(root: TreeNode):
    """
    222.完全二叉树的节点个数
    完全二叉樹：当其每一个结点都与深度为k的满二叉树中编号从1至n的结点
    完全二叉树只有两种情况，情况一：就是满二叉树，情况二：最后一层叶子节点没有满。

    解法：使用普通二叉樹解法
    """
    if root is None: return 0
    leftNum = getNodeNum1(root.left)
    rightNum = getNodeNum1(root.right)
    totalNum = leftNum + rightNum + 1
    return totalNum


def getNodeNum2(root: TreeNode):
    """
    222.完全二叉树的节点个数
    完全二叉樹：当其每一个结点都与深度为k的满二叉树中编号从1至n的结点
    完全二叉树只有两种情况，情况一：就是满二叉树，情况二：最后一层叶子节点没有满。

    解法：使用完全二叉樹特性
    完全二叉树只有两种情况，情况一：就是满二叉树，情况二：最后一层叶子节点没有满。
    对于情况一，可以直接用 2^树深度 - 1 来计算，注意这里根节点深度为1。
    对于情况二，分别递归左孩子，和右孩子，递归到某一深度一定会有左孩子或者右孩子为满二叉树，然后依然可以按照情况1来计算。
    """
    if root is None: return 0
    left = root.left
    right = root.right
    leftHeight = 0
    rightHeight = 0
    while left:
        left = left.left
        leftHeight += 1
    while right:
        right = right.right
        rightHeight += 1
    # 若該樹不為滿二叉樹，則最左下點與最右下點高度必不一致
    if leftHeight == rightHeight:  # 情況一
        return (2 << leftHeight) - 1
    return getNodeNum2(root.left) + getNodeNum2(root.right) + 1  # 情況二


def hasPathSum(root: TreeNode, targetSum: int) -> bool:
    """
    112. 路径总和
    给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。
    """
    def isornot(root: TreeNode, targetSum: int):
        if (root.left is None) and (root.right is None) and targetSum == 0:  # leaf
            return True
        if (root.left is None) and (root.right is None):  # leaf & path sum != target
            return False
        if root.left is not None:
            targetSum -= root.left.val
            if isornot(root.left, targetSum): return True
            targetSum += root.left.val  # 回朔
        if root.right is not None:
            targetSum -= root.right.val
            if isornot(root.right, targetSum): return True
            targetSum += root.right.val
        return False

    if root is None:
        return False
    else:
        return isornot(root, targetSum - root.val)


if __name__ == "__main__":
    import time

    print("__test__")
    # nodes = [3, 9, 20, None, None, 15, 7]
    # root = arr2tree(nodes)
    # print(levelOrder(root))

    # print(isSymertric(arr2tree([1, 2, 2, 3, 4, 4, 3])))
    # print(isSymmetric(arr2tree([1, 2, 2, None, 3, None, 3])))

    # print(isSameTree(arr2tree([1, 2, 3]), arr2tree([1, 2, 3])))

    # print(sumOfLeftLeaves(arr2tree([3, 9, 20, None, None, 15, 7])))

    # print(minDepth(arr2tree([3, 9, 20, None, None, 15, 7])))
    start = time.time()
    print(getNodeNum1(arr2tree([1, 2, 3, 4, 5, 6])))
    print(time.time() - start)
    start = time.time()
    print(getNodeNum2(arr2tree([1, 2, 3, 4, 5, 6])))
    print(time.time() - start)
