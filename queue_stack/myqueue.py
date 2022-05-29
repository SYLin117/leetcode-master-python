class MyQueue:
    """
    232.用栈实现队列
    在push数据的时候，只要数据放进输入栈就好，
    但在pop的时候，操作就复杂一些，输出栈如果为空，
    就把进栈数据全部导入进来（注意是全部导入），
    再从出栈弹出数据，如果输出栈不为空，则直接从出栈弹出数据就可以了。
    """

    def __init__(self):
        """
        in負責push, out負責pop
        """
        self.stack_in = []
        self.stack_out = []

    def empty(self) -> bool:
        """
        只要in或者out有元素，说明队列不为空
        """
        return not (self.stack_in or self.stack_out)

    def push(self, x: int) -> None:
        """
        尾巴加入新元素
        """
        self.stack_in.append(x)

    def pop(self) -> int:
        """
        將頭的資料提出
        """
        if self.empty(): return None
        if self.stack_out:  # stack_out還有東西
            return self.stack_out.pop()
        else:
            for i in range(len(self.stack_in)):
                self.stack_out.append(self.stack_in.pop())
            return self.stack_out.pop()

    def peek(self) -> int:
        """
        get the front element
        """
        ans = self.pop()
        self.stack_out.append(ans)
        return ans



if __name__ == '__main__':
    myQueue = MyQueue()
    myQueue.push(1)
    myQueue.push(2)
    assert 1 == myQueue.pop()
    assert 2 == myQueue.pop()
    myQueue.push(3)
    myQueue.push(4)
    myQueue.push(5)
    assert 3 == myQueue.pop()
