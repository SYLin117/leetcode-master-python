class Solution:
    def fib(self, n: int):
        """
        509. 斐波那契数
        """
        if n == 1 or n == 2:
            return 1
        else:
            return self.fib(n - 1) + self.fib(n - 2)

    def climbStairs(self, n: int):
        """
        70. 爬楼梯
        每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
        解：每階可以是前一階or前二階爬上來
        """
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2, (n + 1)):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

    def minCostClimbingStairs(self, cost: list[int]):
        """
        746. 使用最小花费爬楼梯
        每当你爬上一个阶梯你都要花费对应的体力值，一旦支付了相应的体力值，你就可以选择向上爬一个阶梯或者爬两个阶梯。
        可以选择从下标为 0 或 1 的元素作为初始阶梯。
        输入：cost = [10, 15, 20] 输出：15 解释：最低花费是从 cost[1] 开始，然后走两步即可到阶梯顶，一共花费 15 。
        注意：頂部是len(cost)+1, len(cost)=3, 最頂是第四接 ([10,15,20]0)
        """
        dp = [0] * (len(cost))  ## 這裡的思維要用逆推 dp[i]表示離開i階所需要的體力
        dp[0] = cost[0]
        dp[1] = cost[1]
        for i in range(2, len(cost)):
            dp[i] = min(dp[i - 1], dp[i - 2]) + cost[i]  # 只有i-1, i-2可到達第i階, 所以離開第i階就是dp[i-1] or dp[i-2] + cost[i]
        return min(dp[len(cost) - 1], dp[len(cost) - 2])

    def uniquePaths(self, m: int, n: int):
        """
        62. 不同路徑
        一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）
        机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
        问总共有多少条不同的路径？
        """
        dp = [[1 for i in range(n)] for j in range(m)]  # 建立m*n的list, dp[i][j]表示到達i,j的路徑
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]

    def uniquePaths2(self, obstacleGrid: list[list[int]]):
        """
        63. uniquePaths相同 只是有障礙
        """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[0 for i in range(m)] for j in range(n)]
        dp[0][0] = 1 if obstacleGrid[0][0] != 1 else 0
        for i in range(1, n):  # 第一列
            if dp[0][i - 1] != 0 and obstacleGrid[0][i] != 1:
                dp[0][i] = 1
        for i in range(1, m):
            if dp[i - 1][0] != 0 and obstacleGrid[i][0] != 1:
                dp[i][0] = 1

        for i in range(1, m):  # 注意這裡要從1開始
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]

    def numTrees(self, n: int) -> int:
        """
        96. 不同的二叉搜索树
        给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种？
        解法：
        numTree[4] = numTree[0] * numTree[3] +
                     numTree[1] * numTree[2] +
                     numTree[2] * numTree[1] +
                     numTree[3] * numTree[0]
        """
        numTree = [1] * (n + 1)
        # 0 node = 1 tree
        # 1 node = 1 tree
        for nodes in range(2, n + 1):  # 總node數
            total = 0
            for root in range(1, nodes + 1):  # 總node中用地幾個node做為root(root表示第幾個node, eg. root=2時表示在nodes中第2的點)
                left = root - 1  # 左Node數
                right = nodes - root
                total += numTree[left] * numTree[right]
            numTree[nodes] = total
        return numTree[n]


if __name__ == "__main__":
    sol = Solution()
    # print(sol.fib(4))

    # print(sol.climbStairs())
    # print(sol.minCostClimbingStairs([10, 15, 20]))
    # print(sol.uniquePaths(3, 7))
    print(sol.uniquePaths2([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
