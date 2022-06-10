from treenode import *


class Solution:
    def levelOrder(self, root: TreeNode) -> list:
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

    def invertTree(self, root: TreeNode):
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

    def isSymmetric(self, root: TreeNode):
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

    def binaryTreePaths(self, root: TreeNode) -> list:
        """
        257. 二叉树的所有路径
        给定一个二叉树，返回所有从根节点到叶子节点的路径。
        说明: 叶子节点是指没有子节点的节点。

        解法：递归法+隐形回溯
        """

        def traversal(cur: TreeNode, path: str, result: List[str]) -> None:
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

    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        """
        404.左叶子之和
        分辨左leaf：如果左节点不为空，且左节点没有左右孩子，那么这个节点的左节点就是左叶子
        """
        if not root:
            return 0
        leftValue = self.sumOfLeftLeaves(root.left) if root.left else 0
        rightValue = self.sumOfLeftLeaves(root.right) if root.right else 0

        midValue = 0
        if root.left is not None and root.left.left is None and root.left.right is None:  # 存在左節點存在且左節點沒有子節點
            midValue = root.left.val
        sum = leftValue + rightValue + midValue
        return sum

    def maxDepth(self, root: TreeNode):
        """
        104.二叉树的最大深度
        给定一个二叉树，找出其最大深度。
        二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
        """
        if root is None:
            return 0
        leftDepth = self.maxDepth(root.left)
        rightDepth = self.maxDepth(root.right)
        return 1 + max(leftDepth, rightDepth)

    def minDepth(self, root: TreeNode):
        """
        111.二叉树的最小深度
        最小深度是从根节点到最近叶子节点的最短路径上的节点数量。，注意是叶子节点。

        注意：不可與maxDepth相同
        """
        if root is None: return 0
        leftDepth = self.minDepth(root.left)
        rightDepth = self.minDepth(root.right)

        if (root.left is None and root.right is not None):  # 只有右子樹
            return 1 + rightDepth

        if (root.right is None and root.left is not None):  # 只有左子樹
            return 1 + leftDepth

        result = 1 + min(leftDepth, rightDepth)
        return result

    def getNodeNum1(self, root: TreeNode):
        """
        222.完全二叉树的节点个数
        完全二叉樹：当其每一个结点都与深度为k的满二叉树中编号从1至n的结点
        完全二叉树只有两种情况，情况一：就是满二叉树，情况二：最后一层叶子节点没有满。

        解法：使用普通二叉樹解法
        """
        if root is None: return 0
        leftNum = self.getNodeNum1(root.left)
        rightNum = self.getNodeNum1(root.right)
        totalNum = leftNum + rightNum + 1
        return totalNum

    def getNodeNum2(self, root: TreeNode):
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
        return self.getNodeNum2(root.left) + self.getNodeNum2(root.right) + 1  # 情況二

    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
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

    def hasPathSumII(self, root: TreeNode, targetSum: int):
        """
        113 路径总和-ii
        與112 路徑總和相似 但是回傳符合條件的所有路徑

        解法：遞迴
        """

        def traversal(cur_node, remain):
            if cur_node.left is None and cur_node.right is None and remain == 0:  # 到達leaf且remain為0
                result.append(path[:])
                return
            if cur_node.left is not None:
                path.append(cur_node.left.val)
                traversal(cur_node.left, remain - cur_node.left.val)
                path.pop(-1)
            if cur_node.right is not None:
                path.append(cur_node.right.val)
                traversal(cur_node.right, remain - cur_node.right.val)
                path.pop(-1)

        result, path = [], []
        if root is None:
            return result
        else:
            path.append(root.val)
            traversal(root, targetSum - root.val)
            return result

    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        """
        105. Construct Binary Tree from Preorder and Inorder Traversal
        使用 前序與中序建立二叉樹
        """
        if not preorder:
            return None

        root_val = preorder[0]
        root = TreeNode(root_val)

        # 分割inorder
        seperate_idx = inorder.index(root_val)
        inorder_left = inorder[:seperate_idx]
        inorder_right = inorder[seperate_idx + 1:]

        # 分割preorder
        preorder_left = preorder[1:len(inorder_left) + 1]
        preorder_right = preorder[1 + len(inorder_left):]

        # 遞迴
        root.left = self.buildTree(preorder_left, inorder_left)
        root.right = self.buildTree(preorder_right, inorder_right)
        return root

    def buildMaxTree(self, nums: list):
        """
        654.最大二叉树
        定義：
        根节点：数组中的最大值。
        左子树：最大值左边的数组对应的最大二叉树。
        右子树：最大值右边的数组对应的最大二叉树。
        """

    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
        """
        617.合并二叉树

        """
        newNode = None
        if root1 is not None and root2 is not None:
            newNode = TreeNode(root1.val + root2.val)
        elif root1 is None and root2 is not None:
            newNode = TreeNode(root2.val)
        elif root1 is not None and root2 is None:
            newNode = TreeNode(root1.val)
        else:
            return
        root1_left = root1.left if root1 else None
        root1_right = root1.right if root1 else None
        root2_left = root2.left if root2 else None
        root2_right = root2.right if root2 else None
        if root1_left or root2_left:
            newNode.left = self.mergeTrees(root1_left, root2_left)
        if root1_right or root2_right:
            newNode.right = self.mergeTrees(root1_right, root2_right)

        return newNode

    def searchBST(self, root: TreeNode, val: int):
        """
        700.二叉搜索树中的搜索
        """
        if not root or root.val == val:
            return root
        if val < root.val:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)

    def isValidBST(self, root: TreeNode, ) -> bool:
        """
        98. 驗證ＢＳＴ
        """

        def __isValidBST(root: TreeNode):
            nonlocal cur_max
            if not root:
                return True

            leftValid = __isValidBST(root.left)  # 此步驟會更新cur_max
            if cur_max < root.val:
                cur_max = root.val
            else:
                return False
            rightValid = __isValidBST(root.right)  # 此步驟會更新cur_max

            return leftValid and rightValid

        cur_max = -float("INF")
        return __isValidBST(root)

    def getMinDiff(self, root: TreeNode):
        """
        530.二叉搜索树的最小绝对差

        解法：
        ＢＳＴ在中序的情況相當於有序列表
        """

        def buildList(root):
            """
            使用中序建立list
            """
            if not root:
                return None
            if root.left: buildList(root.left)
            sortList.append(root.val)
            if root.right: buildList(root.right)

        sortList = []
        result = float("inf")
        buildList(root)
        for i in range(1, len(sortList)):
            result = min(result, sortList[i] - sortList[i - 1])
        return result

    def findMode(self, root: TreeNode):
        """
        501.二叉搜索树中的众数

        解法：暴力法
        """

        def searchBST(root: TreeNode):
            nonlocal max_count
            if not root: return
            if root.left: searchBST(root.left)
            if root.val in freq:
                freq[root.val] += 1
                if freq[root.val] > max_count:
                    max_count = freq[root.val]
            else:
                freq[root.val] = 1
                if max_count < 1:
                    max_count = 1
            if root.right: searchBST(root.right)

        max_count = 0
        freq = {}
        searchBST(root)
        min_freq = float('-inf')
        result = []
        for k, v in freq.items():
            if v == max_count:
                result.append(k)
        return result

    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode):
        """
        236. 二叉树的最近公共祖先
        概念：
        如果找到一个节点，发现左子树出现结点p，右子树出现节点q，或者 左子树出现结点q，右子树出现节点p，那么该节点就是节点p和q的最近公共祖先。
        容易忽略一个情况，就是节点本身p(q)，它拥有一个子孙节点q(p)。(即：其中一點是另外一點的祖先)

        解法：
        使用后序遍历，回溯的过程，就是从低向上遍历节点，一旦发现满足第一种情况的节点，就是最近公共节点了。
        但是如果p或者q本身就是最近公共祖先呢？其实只需要找到一个节点是p或者q的时候，直接返回当前节点，无需继续递归子树。
        如果接下来的遍历中找到了后继节点满足第一种情况则修改返回值为后继节点，否则，继续返回已找到的节点即可。
        """
        if not root or root == p or root == q:
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right:
            return root
        if left:
            return left
        return right


if __name__ == "__main__":
    import time

    print("__test__")
    sol = Solution()
    # nodes = [3, 9, 20, None, None, 15, 7]
    # root = arr2tree(nodes)
    # print(levelOrder(root))

    # print(isSymertric(arr2tree([1, 2, 2, 3, 4, 4, 3])))
    # print(isSymmetric(arr2tree([1, 2, 2, None, 3, None, 3])))

    # print(isSameTree(arr2tree([1, 2, 3]), arr2tree([1, 2, 3])))

    # print(sumOfLeftLeaves(arr2tree([3, 9, 20, None, None, 15, 7])))

    # print(minDepth(arr2tree([3, 9, 20, None, None, 15, 7])))
    # start = time.time()
    # print(getNodeNum1(arr2tree([1, 2, 3, 4, 5, 6])))
    # print(time.time() - start)
    # start = time.time()
    # print(getNodeNum2(arr2tree([1, 2, 3, 4, 5, 6])))
    # print(time.time() - start)

    # print(hasPathSumII(arr2tree([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, 5, 1]), 22))

    # root = buildTree(arr2tree([3, 9, 20, 15, 7]), arr2tree([9, 3, 15, 20, 7]))
    # print(root)

    # print2D(arr2tree([1, 3, 2, 5]))
    # print2D(arr2tree([2, 1, 3, None, 4, None, 7]))

    # print2D(arr2tree([2, 1, 3, None, 4, None, 7]))
    # print2D(mergeTrees(arr2tree([1, 3, 2, 5]), arr2tree([2, 1, 3, None, 4, None, 7])))
    # print(sol.searchBTS(arr2tree([4, 2, 7, 1, 3]), val=2))

    # print(sol.isValidBST(arr2tree([2, 1, 3])))

    # print(sol.getMinDiff(arr2tree([1, None, 2])))

    # print2D(arr2tree([1, None, 2, None, None, 2]))
    # print(sol.findMode(arr2tree([1, None, 2, None, None, 2])))
    print(sol.findMode(arr2tree([0])))
