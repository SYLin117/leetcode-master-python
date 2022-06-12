class Solution:
    def combine(self, n: int, k: int) -> list[list[int]]:
        """
        77. 组合
        给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
        示例:
        输入: n = 4, k = 2
        输出:
        [[2,4],[3,4],[2,3],[1,2],[1,3],[1,4],]
        """
        res = []
        path = []

        def backtracking(n, k, startIndex):
            if len(path) == k:
                res.append(path[:])
                return
            for i in range(startIndex, n + 1):
                path.append(i)
                backtracking(n, k, i + 1)
                path.pop()

        backtracking(n, k, 1)  # 因為數字是從1開始
        return res

    def combineOPT(self, n: int, k: int) -> list[list[int]]:
        """
        透過剪枝優化
        """
        res = []  # 存放符合条件结果的集合
        path = []  # 用来存放符合条件结果

        def backtrack(n, k, startIndex):
            if len(path) == k:
                res.append(path[:])
                return
            # 优化的地方，+2是因為 startIndex ~ n-(k-len(path))+1
            # eg. 當n = 4，k = 3, n - (k - 0) + 1 即 4 - ( 3 - 0) + 1 = 2，i最大可以到2 ex. [2,3,4]
            for i in range(startIndex, n - (k - len(path)) + 2):
                path.append(i)  # 处理节点
                backtrack(n, k, i + 1)  # 递归
                path.pop()  # 回溯，撤销处理的节点

        backtrack(n, k, 1)
        return res

    def combinationSum(self, candidates: list[int], target: int):
        """
        39. 组合总和
        所有数字（包括 target）都是正整数。
        解集不能包含重复的组合。
        candidates 中的数字可以无限制重复被选取。
        """
        res = []
        path = []

        def backtrack(sum, startIndex):
            if sum == target:
                res.append(path[:])  # copy path
            if sum > target:
                return
            for i in range(startIndex, len(candidates)):
                sum += candidates[i]
                path.append(candidates[i])
                backtrack(sum, i)  # 因為可以重複 所以backtract startIndex不會+1
                sum -= candidates[i]
                path.pop()

        backtrack(0, 0)
        return res

    def combinationSum2(self, candidates: list[int], target: int):
        """
        40. 組合總和II
        给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
        candidates 中的每个数字在每个组合中只能使用一次。

        範例: [2,5,2,1,2], 5
        """
        res = []
        path = []
        used = [False] * len(candidates)
        # 先排序 避免重複
        candidates.sort()

        def backtrack(sum: int, startIndex: int):
            if sum == target:
                res.append(path[:])
                return
            for i in range(startIndex, len(candidates)):
                if sum + candidates[i] > target:  # 剪枝
                    return
                # 去重 (不可以有重複的組合)
                # 检查同一树层(for loop)是否出现曾经使用过的相同元素
                # 若数组中前后元素值相同，但前者却未被使用(used == False)，说明是for loop中的同一树层的相同元素情况
                if i > 0 and candidates[i] == candidates[i - 1] and used[i - 1] == False:
                    continue
                sum += candidates[i]
                path.append(candidates[i])
                used[i] = True
                backtrack(sum, i + 1)
                used[i] = False
                path.pop()
                sum -= candidates[i]

        backtrack(0, 0)
        return res

    def combinationSum3(self, k: int, n: int):
        """
        216. 组合总和
        找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。
        eg.  k = 3, n = 7 输出: [[1,2,4]]
        """
        res = []
        path = []

        def backtrack(k, n, sumNow: int, startIndex: int):
            if sumNow > n:
                return
            if len(path) == k:  # 只要path==k都要return
                if sumNow == n:
                    res.append(path[:])  # 要拷貝一份path，不然儲存的是reference
                return
            for i in range(startIndex, 10 - (k - len(path)) + 1):
                path.append(i)
                # sumNow += i
                backtrack(k, n, sumNow + i, i + 1)
                path.pop()
                # sumNow -= i

        backtrack(k, n, 0, 1)
        return res

    def partition(self, s: str):
        """
        131. 分割回文子串
        1.切割问题，有不同的切割方式
        2.判断回文
        """
        res = []
        path = []

        def backtrack(startIndex: int):
            if startIndex >= len(s):
                res.append(path[:])
                return
            for i in range(startIndex, len(s)):
                tmp = s[startIndex:i + 1]
                if tmp == tmp[::-1]:  # tmp[::-1]表示reverse
                    path.append(tmp)
                    backtrack(i + 1)
                    path.pop()
                else:
                    continue

        backtrack(0)
        return res

    def restoreIpAddresses(self, s: str):
        """
        93. 回復IP位址
        给定一个只包含数字的字符串，复原它并返回所有可能的 IP 地址格式。
        有效的 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。
        例如："0.1.2.201" 和 "192.168.1.1" 是 有效的 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效的 IP 地址。

        eg.
        输入：s = "25525511135"
        输出：["255.255.11.135","255.255.111.35"]
        """
        res = []

        def isValid(s, start, end):  # start, end表示index (end有包含)
            # 檢查數字是否合法
            if start > end: return False
            if s[start] == '0' and start != end:  # 不可以有0起頭的數字
                return False
            if not 0 <= int(s[start:end + 1]) <= 255:  # 需介於0~255
                return False
            return True

        def backtrack(s, startIndex, pointNum):  # pointNum紀錄逗點數量
            if pointNum == 3:  # 已經取了三段
                if isValid(s, startIndex, len(s) - 1):  # 剩下的段落也合法
                    res.append(s[:])
                return
            for i in range(startIndex, len(s)):
                # 節取s[startIndex:i]
                if isValid(s, startIndex, i):
                    s = s[:i + 1] + '.' + s[i + 1:]  # 將s切半
                    backtrack(s, i + 2, pointNum + 1)  # +2因為中間有插入'.'
                    s = s[:i + 1] + s[i + 2:]  # 回復
                else:  # 超過255就不需要loop了
                    break

        backtrack(s, 0, 0)
        return res

    def permute(self, nums: list[int]):
        """
        46. 全排列
        给定一个 没有重复 数字的序列，返回其所有可能的全排列。
        输入: [1,2,3]
        输出: [ [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1] ]
        """
        res = []
        path = []
        used = [False] * len(nums)

        def backtrack():
            if len(path) == len(nums):
                res.append(path[:])
                return
            for i in range(0, len(nums)):  # 每層都會把所有數loop
                if used[i] == True:  # 跳過已經使用的數
                    continue
                used[i] = True
                path.append(nums[i])
                backtrack()
                path.pop()
                used[i] = False

        backtrack()
        return res

    def permute2(self, nums: list[int]):
        """
        46. 全排列
        给定一个 没有重复 数字的序列，返回其所有可能的全排列。
        输入：nums = [1,1,2]
        输出： [[1,1,2], [1,2,1], [2,1,1]]
        """
        res = []
        path = []
        used = [False] * len(nums)
        nums.sort()

        def backtrack():
            if len(path) == len(nums):
                res.append(path[:])
                return
            for i in range(0, len(nums)): # loop 所有數
                if used[i] == True:  # 跳過已經使用的數(每個數不可重複使用)
                    continue
                if i > 0 and nums[i] == nums[i-1] and used[i - 1]:  # 跳過重複的情況(當nums中有重複的數)
                    continue
                used[i] = True
                path.append(nums[i])
                backtrack()
                path.pop()
                used[i] = False

        backtrack()
        return res


if __name__ == "__main__":
    sol = Solution()
    # print(sol.combine(4, 2))

    # print(sol.combinationSum([2, 3, 5], 8))
    # print(sol.combinationSum2([2, 5, 2, 1, 2], 5))
    # print(sol.combinationSum3(3, 7, ))

    # print(sol.partition("aab"))

    # print(sol.restoreIpAddresses('0000'))

    # print(sol.permute([1, 2, 3]))
    print(sol.permute2([1, 1, 2]))
