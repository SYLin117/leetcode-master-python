def reverseString(s: str) -> None:
    """
    344.反转字符串
    modify inplace
    """
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1


def reverseStr(s: str, k: int) -> str:
    """
    541. 反转字符串II
    给定一个字符串 s 和一个整数 k，你需要对从字符串开头算起的每隔 2k 个字符的前 k 个字符进行反转。
    如果剩余字符少于 k 个，则将剩余字符全部反转。
    如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。
    eg.
    输入: s = "abcdefg", k = 2
    输出: "bacdfeg"
    解法：
    1. 使用range(start, end, step)来确定需要调换的初始位置
    2. 对于字符串s = 'abc'，如果使用s[0:999] ===> 'abc'。字符串末尾如果超过最大长度，则会返回至字符串最后一个值，这个特性可以避免一些边界条件的处理。
    3. 用切片整体替换，而不是一个个替换.
    """

    def reverse_substring(text):
        left, right = 0, len(text) - 1
        while left < right:
            text[left], text[right] = text[right], text[left]
            left += 1
            right -= 1
        return text

    s = list(s)
    for cur in range(0, len(s), 2 * k):
        s[cur: cur + k] = reverse_substring(s[cur:cur + k])

    return ''.join(s)


def replace_space(s: str):
    """
    剑指Offer 05.替换空格
    """
    counter = s.count(' ')  # 計算空白的次數

    res = list(s)
    # 每碰到一个空格就多拓展两个格子，1 + 2 = 3个位置存’%20‘
    res.extend([' '] * counter * 2)

    # 原始字符串的末尾，拓展后的末尾
    left, right = len(s) - 1, len(res) - 1

    while left >= 0:
        if res[left] != ' ':
            res[right] = res[left]
            right -= 1
        else:
            # [right - 2, right), 左闭右开
            res[right - 2: right + 1] = '%20'
            right -= 3
        left -= 1
    return ''.join(res)


def reverse_words(s: str) -> str:
    """
    151.翻转字符串里的单词
    给定一个字符串，逐个翻转字符串中的每个单词。
    示例 1：
    输入: "the sky is blue"
    输出: "blue is sky the"
    示例 2：
    输入: "  hello world!  "
    输出: "world! hello"
    解释: 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
    示例 3：
    输入: "a good   example"
    输出: "example good a"
    解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。

    本方法不使用內建func
    """

    def trim_space(s: str) -> list:
        """
        將多餘的空白去除
        """
        n = len(s)
        left = 0
        right = n - 1
        while left <= right and s[left] == ' ':  # 去除開頭的空格
            left += 1
        while left <= right and s[right] == ' ':  # 去除结尾的空格
            right = right - 1
        tmp = []
        while left <= right:
            if s[left] != ' ':  # 非空白
                tmp.append(s[left])
            elif tmp[-1] != ' ':  # 是空白 但是前一個欄位不是空白(因為若前一個是空白，則tmp最後一個字是空白)
                tmp.append(s[left])
            left += 1
        return tmp

    def reverse_string(s: list, left: int, right: int):
        """
        反轉list
        """
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
        return None

    def reverse_each_word(s: list):
        start = 0  # 每個字的start
        end = 0  # 每個字的end
        n = len(s)
        while start < n:
            while end < n and s[end] != ' ':  # 一直到s[end]為空白為止
                end += 1
            reverse_string(s, start, end-1)
            start = end + 1
            end += 1
        return None

    l = trim_space(s) # 去除多的空白
    reverse_string(l, 0, len(l) - 1) # 反轉整個字串
    reverse_each_word(l) # 反轉每個字
    return ''.join(l)



if __name__ == "__main__":
    # print('test')
    # print(reverseStr("abcdefg", 2))

    # print(replace_space("We are happy."))

    print(reverse_words("a good   example"))
