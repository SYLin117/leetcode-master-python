from collections import deque


class MyStack:
    """
    225. 用队列实现栈
    push(x) -- 元素 x 入栈
    pop() -- 移除栈顶元素
    top() -- 获取栈顶元素
    empty() -- 返回栈是否为空

    只能使用队列的基本操作-- 也就是 push to back, peek/pop from front, size, 和 is empty 这些操作是合法的。
    所使用的语言也许不支持队列。 你可以使用 list 或者 deque（双端队列）来模拟一个队列 , 只要是标准的队列操作即可。
    """

    def __init__(self):
        self.queue_in = deque()
        self.queue_out = deque()

    def push(self, x: int) -> None:
        self.queue_in.append(x)

    def pop(self) -> int:
        if len(self.queue_in) == 0:
            return None
        else:
            for i in range(len(self.queue_in) - 1):  # 將queue_in移到queue_out，留下最後一個
                self.queue_out.append(self.queue_in.pop())

            for i in range(len(self.queue_out)):  # 將queue_out移回queue_in
                self.queue_in.append(self.queue_out.pop())

            return self.queue_in.pop()  # queue_in最前面的就是最後的

    def top(self) -> int:
        ans = self.pop()
        self.queue_in.append(ans)
        return ans

    def empty(self):
        if len(self.queue_in) == 0:
            return True
        else:
            return False


if __name__ == "__main__":
    queue = MyStack()
    queue.push(1)
    queue.push(2)
    assert 2 == queue.pop()  ## 注意弹出的操作
    queue.push(3)
    queue.push(4)
    assert 4 == queue.pop()  ## 注意弹出的操作
    assert 3 == queue.pop()
    assert 1 == queue.pop()
    assert True == queue.empty()
