import collections


def commonChars(words: list[str]) -> list[str]:
    """
    1002. 查找常用字符
    给你一个字符串数组 words ，请你找出所有在 words 的每个字符串中都出现的共用字符（ 包括重复字符），并以数组形式返回。你可以按 任意顺序 返回答案。
    Args:
        words: 輸入字串

    Returns:

    """
    # 取第一個word作為base
    tmp = collections.Counter(words[0])  # Counter可以用來計數
    result = []
    for i in range(1, len(words)):
        tmp = tmp & collections.Counter(words[i])  # 取交集

    for j in tmp:
        v = tmp[j]
        while (v):
            result.append(j)
            v -= 1
    return result


def intersection(nums1: list[int], nums2: list[int]) -> list[int]:
    """
    349. 两个数组的交集
    给定两个数组，编写一个函数来计算它们的交集。
    Args:
        nums1:
        nums2:

    Returns:

    """
    return list(set(nums1), set(nums2))


def isHappy(n: int) -> bool:
    """
    第202题. 快乐数
    编写一个算法来判断一个数 n 是不是快乐数。
    「快乐数」定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。如果 可以变为  1，那么这个数就是快乐数。

    题目中说了会 无限循环，那么也就是说求和的过程中，sum会重复出现，这对解题很重要！
    当我们遇到了要快速判断一个元素是否出现集合里的时候，就要考虑哈希法了。
    """

    def cal_happy(num):
        """
        計算快樂數
        """
        _sum = 0
        while num:
            _sum += (num % 10) ** 2
            num = int(num / 10)
        return _sum

    record = set()
    while True:
        n = cal_happy(n)
        if n == 1:
            return True

        if n in record:
            return False
        else:
            record.add(n)


def twoSum(nums: list[int], target: int) -> list[int]:
    """
    1. 两数之和

    给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
    你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
    """
    rec = {}  # key: 數值, value: index
    for i in range(len(nums)):
        rest = target - nums[i]
        if rec.get(rest, None) is not None: return [rec[rest], i]  # 有配對回傳解
        rec[nums[i]] = i  # 沒有配對 加入map


def fourSumCount(nums1: list, nums2: list, nums3: list, nums4: list) -> int:
    """
    第454题.四数相加II
    给定四个包含整数的数组列表 A , B , C , D ,计算有多少个元组 (i, j, k, l) ，使得 A[i] + B[j] + C[k] + D[l] = 0。

    解法: n1+n2+n3+n4 = 0 所以 n1+n2 = -n3-n4

    範例：
    输入: A = [ 1, 2] B = [-2,-1] C = [-1, 2] D = [ 0, 2] 输出: 2 解释: 两个元组如下:
    (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
    (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0
    """
    hashmap = dict()
    for n1 in nums1:
        for n2 in nums2:
            if n1 + n2 in hashmap:
                hashmap[n1 + n2] += 1
            else:
                hashmap[n1 + n2] = 1
    count = 0
    for n3 in nums3:
        for n4 in nums4:
            key = - n3 - n4
            if key in hashmap:
                count += hashmap[key]
    return count


def canConstruct(ransomNote: str, magazine: str) -> bool:
    """
    383. 赎金信
    给定一个赎金信 (ransom) 字符串和一个杂志(magazine)字符串，判断第一个字符串 ransom 能不能由第二个字符串 magazines 里面的字符构成。如果可以构成，返回 true ；否则返回 false。
    """
    hashmap = dict()
    for s in ransomNote:
        if s in hashmap:
            hashmap[s] += 1
        else:
            hashmap[s] = 1
    for l in magazine:
        if l in hashmap:
            hashmap[l] -= 1
    for key, value in hashmap.items():
        if value > 0:
            return False
    return True


def threeSum(nums: list[int]) -> list:
    """
    第15题. 三数之和
    给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。

    注意：[0， 0， 0， 0] 这组数据

    解法：雙指針法
    依然还是在数组中找到 abc 使得a + b +c =0，我们这里相当于 a = nums[i] b = nums[left] c = nums[right]。
    接下来如何移动left 和right呢， 如果nums[i] + nums[left] + nums[right] > 0 就说明 此时三数之和大了，因为数组是排序后了，所以right下标就应该向左移动，这样才能让三数之和小一些。
    如果 nums[i] + nums[left] + nums[right] < 0 说明 此时 三数之和小了，left 就向右移动，才能让三数之和大一些，直到left与right相遇为止。
    """
    result = []
    if len(nums) < 3: return result
    nums.sort()
    for i in range(len(nums)):
        if nums[i] > 0: break  ## nums以排序 若第一個數就大於0則sum必不可能為0
        if i > 0 and nums[i] == nums[i - 1]:  # 去重 eg.[0,0,0,0]
            continue
        left = i + 1
        right = len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total > 0:
                right -= 1
            elif total < 0:
                left += 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                while left != right and nums[left] == nums[left + 1]: left += 1  # 略過重複的value
                while left != right and nums[right] == nums[right - 1]: right -= 1  # 略過重複的value
                left += 1
                right -= 1
    return result


def fourSum(nums: list[int], target: int) -> list[list[int]]:
    """
    第18题. 四数之和

    题意：给定一个包含 n 个整数的数组 nums 和一个目标值 target，
    判断 nums 中是否存在四个元素 a，b，c 和 d ，
    使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。

    解法：雙指針法
    四数之和，和15.三数之和是一个思路，都是使用双指针法, 基本解法就是在15.三数之和 的基础上再套一层for循环。
    四数之和的双指针解法是两层for循环nums[k] + nums[i]为确定值，
    依然是循环内有left和right下标作为双指针，
    找出nums[k] + nums[i] + nums[left] + nums[right] == target的情况，三数之和的时间复杂度是O(n^2)，四数之和的时间复杂度是O(n^3)。
    """
    nums.sort()
    n = len(nums)
    res = []
    for i in range(n):
        if i > 0 and nums[i] == nums[i - 1]: continue  # 去重
        for j in range(i + 1):
            if j > i + 1 and nums[j] == nums[j - 1]: continue
            p = j + 1
            q = n - 1
            while p < q:
                if nums[i] + nums[j] + nums[p] + nums[q] > target:
                    q -= 1
                elif nums[i] + nums[j] + nums[p] + nums[q] < target:
                    p += 1
                else:
                    res.append([nums[i], nums[j], nums[p], nums[q]])
                    while p < q and nums[p] == nums[p + 1]: p += 1
                    while p < q and nums[q] == nums[q - 1]: p -= 1
                    p += 1
                    q -= 1
    return res


if __name__ == "__main__":
    # words = ['bella', 'label', 'roller']
    # commonChars(words)

    # print(isHappy(19))

    print(threeSum([-1, 0, 1, 2, -1, -4]))
