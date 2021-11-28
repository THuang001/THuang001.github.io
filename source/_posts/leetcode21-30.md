---

title: leetcode21-25
date: 2019-06-20 20:59:12
tags: leetcode
categories: 算法

---

## 1. 合并两个有序链表(Easy)

将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

**示例：**

```
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
```

**解答：**

**思路：时间复杂度: $O(n)$，空间复杂度:$ O(1)**$

同样是**dummy head**

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if not l1:
            return l2
        if not l2:
            return l1
        dummy = cur = ListNode(-1)
        while l1 and l2:
            if l1.val<l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 if l1 else l2
        return dummy.next
```

## 2. 括号生成(Medium)

给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。

```
例如，给出 n = 3，生成结果为：
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
```

**解答：**

**思路一：**

官方解答：**回溯法**

只有在我们知道序列仍然保持有效时才添加 '(' or ')'，而不是每次添加。我们可以通过跟踪到目前为止放置的左括号和右括号的数目来做到这一点，

如果我们还剩一个位置，我们可以开始放一个左括号。 如果它不超过左括号的数量，我们可以放一个右括号。

```python
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        ans = []
        def backtrack(s='',left = 0,right = 0,n = n):
            if len(s) == 2 * n:
                ans.append(s)
                return
            if left < n:
                backtrack(s+'(',left+1,right,n)
            if right < left:
                backtrack(s+')',left,right+1,n)
            
        backtrack()
        return ans
```

以这道题为例，backtrack的题应该怎么去思考？

所谓Backtracking都是这样的思路：在当前局面下，你有若干种选择。那么尝试每一种选择。如果已经发现某种选择肯定不行（因为违反了某些限定条件），就返回；如果某种选择试到最后发现是正确解，就将其加入解集

所以你思考递归题时，只要明确三点就行：选择 (Options)，限制 (Restraints)，结束条件 (Termination)。即“ORT原则”（这个是我自己编的）。

对于这道题，在任何时刻，你都有两种选择：

1. 加左括号。
2. 加右括号。

同时有以下限制：

1. 如果左括号已经用完了，则不能再加左括号了。
2. 如果已经出现的右括号和左括号一样多，则不能再加右括号了。因为那样的话新加入的右括号一定无法匹配。

结束条件是：左右括号都已经用完。

结束后的正确性：

左右括号用完以后，一定是正确解。因为1. 左右括号一样多，2. 每个右括号都一定有与之配对的左括号。因此一旦结束就可以加入解集（有时也可能出现结束以后不一定是正确解的情况，这时要多一步判断）。

递归函数传入参数：

限制和结束条件中有“用完”和“一样多”字样，因此你需要知道左右括号的数目。当然你还需要知道当前局面sublist和解集res。

```python
if (左右括号都已用完) {
  加入解集，返回
}
//否则开始试各种选择
if (还有左括号可以用) {
  加一个左括号，继续递归
}
if (右括号小于左括号) {
  加一个右括号，继续递归
}
```

这题其实是最好的backtracking初学练习之一，因为ORT三者都非常简单明显。你不妨按上述思路再梳理一遍，还有问题的话再说。

以上文字来自 **1point3arces**的牛人解答

### 复杂度分析：

我们的复杂度分析依赖于理解 $generateParenthesis(n)$ 中有多少个元素。这个分析超出了本文的范畴，但事实证明这是第 n 个卡塔兰数 $\dfrac{1}{n+1}\binom{2n}{n} $，这是由 $\dfrac{4^n}{n\sqrt{n}} $ 渐近界定的。

时间复杂度：$O(\dfrac{4^n}{\sqrt{n}})$，在回溯过程中，每个有效序列最多需要 n 步。

空间复杂度：$O(\dfrac{4^n}{\sqrt{n}})$，如上所述，并使用 $O(n)$ 的空间来存储序列。

**思路二：动态规划**

**来自本题题解的解答。**

在此题中，动态规划的思想类似于数学归纳法，当知道所有i<n的情况时，我们可以通过某种算法算出i=n的情况。 本题最核心的思想是，考虑i=n时相比n-1组括号增加的那一组括号的位置。

具体思路如下： 当我们清楚所有i<n时括号的可能生成排列后，对与i=n的情况，我们考虑整个括号排列中最左边的括号。 它一定是一个左括号，那么它可以和它对应的右括号组成一组完整的括号"( )"，我们认为这一组是相比n-1增加进来的括号。

那么，剩下n-1组括号有可能在哪呢？ **剩下的括号要么在这一组新增的括号内部，要么在这一组新增括号的外部（右侧）**。既然知道了i<n的情况，那我们就可以对所有情况进行遍历： **"(" + 【i=p时所有括号的排列组合】 + ")" + 【i=q时所有括号的排列组合】 其中 p + q = n-1，且p q均为非负整数。** 事实上，当上述p从0取到n-1，q从n-1取到0后，所有情况就遍历完了。

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        if n == 0:
            return []
        total_l = []
        total_l.append([None])
        total_l.append(["()"])
        for i in range(2,n+1):  # 开始计算i时的括号组合，记为l
            l = []
            for j in range(i): #遍历所有可能的括号内外组合
                now_list1 = total_l[j]
                now_list2 = total_l[i-1-j]
                for k1 in now_list1:  #开始具体取内外组合的实例
                    for k2 in now_list2:
                        if k1 == None:
                            k1 = ""
                        if k2 == None:
                            k2 = ""
                        el = "(" + k1 + ")" + k2
                        l.append(el)
            total_l.append(l)
        return total_l[n]
```

## 3. 合并K个有序列表(Hard)

合并 k 个排序链表，返回合并后的排序链表。请分析和描述算法的复杂度。

**示例:**

```
输入:
[
  1->4->5,
  1->3->4,
  2->6
]
输出: 1->1->2->3->4->4->5->6
```

**解答：**

这道题有多种解法，都可以轻松理解，重要的是掌握和记住。

**思路一：暴力解法。**

- 遍历所有链表，将所有节点的值放到一个数组中。
- 将这个数组排序，然后遍历所有元素得到正确顺序的值。
- 用遍历得到的值，创建一个新的有序链表。

```python
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        self.nodes = []
        head = point = ListNode(0)
        for l in lists:
            while l:
                self.nodes.append(l.val)
                l = l.next
        for x in sorted(self.nodes):
            point.next = ListNode(x)
            point = point.next
        return head.next
```

### 复杂度分析

时间复杂度：$O(N\log N)$ ，其中 $N$ 是节点的总数目。

1. 遍历所有的值需花费 $O(N)$ 的时间。
2. 一个稳定的排序算法花费 $O(N\log N)$ 的时间。
3. 遍历同时创建新的有序链表花费 $O(N)$ 的时间。

空间复杂度：$O(N)$。

1. 排序花费 $O(N)$ 空间（这取决于你选择的算法）。
2. 创建一个新的链表花费 $O(N)$ 的空间。

**执行用时 :132 ms, 在所有 Python 提交中击败了63.46%的用户**

**思路二：优先队列（必须掌握）**

算法：

- 比较 k 个节点（每个链表的首节点），获得最小值的节点。
- 将选中的节点接在最终有序链表的后面。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        import heapq
        dummy = ListNode(0)
        p = dummy
        head = []
        for i in range(len(lists)):
            if lists[i] :
                heapq.heappush(head, (lists[i].val, i))
                lists[i] = lists[i].next
        while head:
            val, idx = heapq.heappop(head)
            p.next = ListNode(val)
            p = p.next
            if lists[idx]:
                heapq.heappush(head, (lists[idx].val, idx))
                lists[idx] = lists[idx].next
        return dummy.next
```

**执行用时 :116 ms, 在所有 Python 提交中击败了79.42%的用户**

**思路三：分治算法**

类似于归并排序

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists:return 
        n = len(lists)
        return self.merge(lists, 0, n-1)
    def merge(self,lists, left, right):
        if left == right:
            return lists[left]
        mid = left + (right - left) // 2
        l1 = self.merge(lists, left, mid)
        l2 = self.merge(lists, mid+1, right)
        return self.mergeTwoLists(l1, l2)
    def mergeTwoLists(self,l1, l2):
        if not l1:return l2
        if not l2:return l1
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```

**执行用时 :136 ms, 在所有 Python 提交中击败了60.55%的用户**

## 4. 两两交换链表中的节点(Medium)

给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。

你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

**示例:**

```
给定 1->2->3->4, 你应该返回 2->1->4->3.
```

**解答：**

**思路一：递归**

**一眼就看出来要用递归来做。**

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        tmp = head.next
        head.next = self.swapPairs(head.next.next)
        tmp.next = head
        return tmp
```

**执行用时 :20 ms, 在所有 Python 提交中击败了93.45%的用户**

思路二：用**loop**做，and **dummy大法**对于nodeList这类题简直无敌。

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        cur = dummy = ListNode(-1)
        dummy.next = head        
        while cur.next and cur.next.next:
            one,two,three = cur.next,cur.next.next,cur.next.next.next
            cur.next = two
            two.next = one
            one.next = three
            cur = one
        return dummy.next
```

**执行用时 :12 ms, 在所有 Python 提交中击败了100.00%的用户**

## 5. K个一组翻转链表(Hard)

给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。

k 是一个正整数，它的值小于或等于链表的长度。

如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

**示例 :**

```
给定这个链表：1->2->3->4->5

当 k = 2 时，应当返回: 2->1->4->3->5

当 k = 3 时，应当返回: 3->2->1->4->5
```

说明 :你的算法只能使用常数的额外空间。
你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

**解答：**

**思路一：递归版本**

可以递归操作, 有两种情况：

1. 就是压根没有k个node，那么我们直接保持这个k-group不动返回head
2. 如果有k个node的话，那么我们先找到第k个node之后的递归结果 node = nxt，然后反转前面k个node，让反转结果的结尾 tail.next = nxt

也可以这样理解，在解决所有node时，将前k个node和剩余的node分开，剩余的node可以递归调用解决，这样，问题就变成了前k个node链表方向依次反向来解决。

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution(object):
    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        cur = head
        cnt = 0
        while cur and cnt != k:
            cur = cur.next
            cnt += 1
        if cnt == k:
            cur = self.reverseKGroup(cur,k)
            while cnt:
                tmp = head.next
                head.next = cur
                cur = head
                head = tmp
                cnt -= 1
            head = cur
        return head
```

**思路二：用堆栈**

用栈，我们把 k 个数压入栈中，然后弹出来的顺序就是翻转的！

这里要注意几个问题：

1. 第一，剩下的链表个数够不够 k 个（因为不够 k 个不用翻转）；

2. 第二，已经翻转的部分要与剩下链表连接起来

```python
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        dummy = ListNode(0)
        p = dummy
        while True:
            count = k 
            stack = []
            tmp = head
            while count and tmp:
                stack.append(tmp)
                tmp = tmp.next
                count -= 1
            # 注意,目前tmp所在k+1位置
            # 说明剩下的链表不够k个,跳出循环
            if count : 
                p.next = head
                break
            # 翻转操作
            while stack:
                p.next = stack.pop()
                p = p.next
            #与剩下链表连接起来 
            p.next = tmp
            head = tmp        
        return dummy.next
```

**思路三：尾插法**

见[本题题解](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/solution/kge-yi-zu-fan-zhuan-lian-biao-by-powcai/)