def isValidBracket(s: str) -> bool:
    """
    20. 有效的括号
    檢視字串中的括號是否配對

    eg.
    1.
    输入: "()[]{}"
    输出: true
    2.
    输入: "([)]"
    输出: false
    """
    stack = []
    for item in s:
        if item == '(':
            stack.append(')')
        elif item == '[':
            stack.append(']')
        elif item == '{':
            stack.append('}')
        elif not stack or stack[-1] != item:  # stack empty or stack[-1]不等於又括號(stack[-1]相當於peek)
            return False
        else:
            stack.pop()
    return True if not stack else False  # stack empty 則 True


def remove_duplicates(s: str) -> str:
    """
    1047. 删除字符串中的所有相邻重复项

    输入："abbaca"
     输出："ca"
     解释：例如，在 "abbaca" 中，我们可以删除 "bb" 由于两字母相邻且相同，这是此时唯一可以执行删除操作的重复项。之后我们得到字符串 "aaca"，其中又只有 "aa" 可以执行重复项删除操作，所以最后的字符串为 "ca"。
    """
    stack = []
    for ch in s:
        if not stack or ch != stack[-1]:
            stack.append(ch)
        else:
            stack.pop()
    return "".join(stack)


def eval_rpc(tokens: list) -> int:
    """
    150. 逆波兰表达式求值 (逆波兰表达式：是一种后缀表达式，所谓后缀就是指算符写在后面。)
    根据 逆波兰表示法，求表达式的值。

    eg.
    输入: ["2", "1", "+", "3", " * "]
    输出: 9
    解释: 该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9
    """
    stack = []
    for token in tokens:
        if token == '+' or token == '-' or token == '*' or token == '/':
            num2 = stack.pop()
            num1 = stack.pop()
            if token == '+':
                stack.append(num1 + num2)
            elif token == '-':
                stack.append(num1 - num2)
            elif token == '*':
                stack.append(num1 * num2)
            elif token == '/':
                stack.append(int(num1 / num2))
        else:
            stack.append(int(token))
    result = stack.pop()
    return result


def max_sliding_window(nums: list, k: int) -> list:
    """
    239. 滑动窗口最大值
    给定一个数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
    """

    class MyQueue:
        """
        透過自製Queue來達成
        """

        def __init__(self):
            self.queue = []

        def pop(self, value):
            # 當要剃除的value還在queue裡面才會替除
            if self.queue and value == self.queue[0]:
                self.queue.pop(0)  # 因為push的設定，最大值永遠會在第一位

        def push(self, value):
            # 若新增的value大於queue中最後一個value，則將前面的value都pop出去(確保最大值在queue.front[0](第一位))
            while self.queue and self.queue[-1] < value:
                self.queue.pop()
            self.queue.append(value)

        def front(self):
            return self.queue[0]

    queue = MyQueue()
    result = []
    for i in range(k):
        queue.push(nums[i])
    result.append(queue.front())
    for i in range(k, len(nums)):
        queue.pop(nums[i - k])  # 當要剃除的value還在queue裡面才會替除
        queue.push(nums[i])  # 加入新value
        result.append(queue.front())
    return result


def top_k_frequency(nums: list[int], k: int) -> list[int]:
    """
    347.前 K 个高频元素
    给定一个非空的整数数组，返回其中出现频率前 k 高的元素。

    解法：
    使用priority queue
    """
    import heapq
    map_ = {}
    for num in nums:  # 計算頻率
        map_[num] = map_.get(num, 0) + 1

    # 对频率排序
    # 定义一个小顶堆，大小为k
    pri_que = []  # 小顶堆 # Heap（堆積）是一顆二元樹，樹上所有父節點的值都小於等於他的子節點的值。

    # 用固定大小为k的小顶堆，扫面所有频率的数值
    for key, freq in map_.items():
        heapq.heappush(pri_que, (freq, key))  # 把 item 放進 heap，並保持 heap 性質不變。
        if len(pri_que) > k:  # 如果堆的大小大于了K，则队列弹出，保证堆的大小一直为k
            heapq.heappop(pri_que)  # 從 heap 取出並回傳最小的元素

    # 找出前K个高频元素，因为小顶堆先弹出的是最小的，所以倒序来输出到数组
    result = [0] * k
    for i in range(k - 1, -1, -1):  # 從 (k-1)~0
        result[i] = heapq.heappop(pri_que)[1]  # 取出key
    return result


def simplifyPath(path: str) -> str:
    """
    71. Simplify Path
    簡化路徑

    eg.
    1. /home/ -> /home
    2. /../ -> /
    3. /home//foo/ -> /home/foo


    解法：
    使用stack
    """
    stack = []
    cur = ''
    for c in path + '/':
        if c == '/':  # 遇到/時
            if cur == '..':  # 若cur==.. 清出stack最後一個元素
                if stack: stack.pop()
            elif cur != "" and cur != ".":  # cur不是空白與..，加入stack
                stack.append(cur)
            cur = ""  # 因為每次遇到/都會清空cur，所以不會有//的情況
        else:  # not /
            cur += c
    return "/" + "/".join(stack)


if __name__ == "__main__":
    # print(eval_rpc(["4", "13", "5", "/", "+"]))

    # print(max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], k=3))

    # print(top_k_frequency([1, 1, 1, 2, 2, 3], 2))

    print(simplifyPath('/home//foo/'))
