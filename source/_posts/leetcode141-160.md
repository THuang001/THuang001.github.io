---
title: leetcode141-160
date: 2019-07-21 20:45:22
tags: leetcode
categories: 算法
---

## 1. 环形链表(Easy)

给定一个链表，判断链表中是否有环。

为了表示给定链表中的环，我们使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 `pos` 是 `-1`，则在该链表中没有环。

 

**示例 1：**

```
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。
```

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png)

**示例 2：**

```
输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。
```

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test2.png)

**示例 3：**

```
输入：head = [1], pos = -1
输出：false
解释：链表中没有环。
```

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test3.png)

 

**进阶：**

你能用 *O(1)*（即，常量）内存解决此问题吗？

解答：

思路：

思路一，快慢指针，即双指针法

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if  not head or not head.next:
            return False
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
```

另外，链表置空法

运行速度比双指针慢。

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head or not head.next:
            return False
        while head.next and head.val != None:
            head.val = None
            head = head.next
        if not head.next:
            return False
        else:
            return True
```



## 2. 环形链表2(Medium)

给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 `null`。

为了表示给定链表中的环，我们使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 `pos` 是 `-1`，则在该链表中没有环。

**说明：**不允许修改给定的链表。

 

**示例 1：**

```
输入：head = [3,2,0,-4], pos = 1
输出：tail connects to node index 1
解释：链表中有一个环，其尾部连接到第二个节点。
```

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png)

**示例 2：**

```
输入：head = [1,2], pos = 0
输出：tail connects to node index 0
解释：链表中有一个环，其尾部连接到第一个节点。
```

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test2.png)

**示例 3：**

```
输入：head = [1], pos = -1
输出：no cycle
解释：链表中没有环。
```

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test3.png)

 

**进阶：**
你是否可以不用额外空间解决此题？

解答：

思路：

这道题和上述题思路一样

* 快慢指针找到环中的节点，然后找到环的入口
* 设两指针fast,slow指向链表头部head，fast每轮走2步，slow每轮走一步：
  * fast指针走到链表末端，说明无环，返回null；
  * 当fast == slow时跳出迭代break；
  * 设两指针分别走了f,s步，设链表头部到环需要走a步，链表环长度b步，则有：
    * 快指针走了满指针两倍的路程f = 2s，快指针比慢指针多走了n个环的长度f = s + nb（因为每走一轮，fast与slow之间距离+1，如果有环快慢指针终会相遇）；
    * 因此可推出：f = 2nb,s = nb，即两指针分别走了2n个环、n个环的周长。

* 接下来，我们将fast指针重新指向头部，并和slow指针一起向前走，每轮走一步，则有：
当fast指针走了a步时，slow指针正好走了a + nb步，此时两指针同时指向链表环入口。
* 最终返回fast。

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return None
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        else:
            return None
        while head != slow:
            slow = slow.next
            head = head.next
        return head
```



## 3. 重排链表(Medium)

给定一个单链表 *L*：*L*0→*L*1→…→*L**n*-1→*L*n ，
将其重新排列后变为： *L*0→*L**n*→*L*1→*L**n*-1→*L*2→*L**n*-2→…

你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

**示例 1:**

```
给定链表 1->2->3->4, 重新排列为 1->4->2->3.
```

**示例 2:**

```
给定链表 1->2->3->4->5, 重新排列为 1->5->2->4->3.
```

解答：

思路：

双指针思路

快慢指针找到中点

然后将链表一分为二

后面的链表倒序

然后合并两个链表

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
        if not head or not head.next:
            return 
        s = f = head
        while f and f.next:
            s = s.next
            f = f.next.next
        head2 = s.next
        s.next = None
        p = None
        cur = head2
        while cur:
            tmp = cur.next
            cur.next = p
            p = cur
            cur = tmp
        l1 = head
        l2 = p
        while l1 and l2:
            tmp = l2.next
            l2.next = l1.next
            l1.next = l2
            l1 = l1.next.next
            l2 = tmp
```



## 4. 二叉树的前序遍历(Medium)

给定一个二叉树，返回它的 *前序* 遍历。

 **示例:**

```
输入: [1,null,2,3]  
   1
    \
     2
    /
   3 

输出: [1,2,3]
```

**进阶:** 递归算法很简单，你可以通过迭代算法完成吗？

解答：

思路：

很简单，递归和迭代都很简单。

迭代

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        res = []
        node = root
        stack = []
        while node or len(stack)>0:
            while node:
                stack.append(node)
                res.append(node.val)
                node = node.left
            if len(stack)>0:
                node = stack.pop()
                node = node.right
        return res
```

递归

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        res = []
        res.append(root.val)
        if root.left:
            res+= (self.preorderTraversal(root.left))
        if root.right:
            res+= (self.preorderTraversal(root.right))
```



## 5. 二叉树的后序遍历(Medium)

给定一个二叉树，返回它的 *后序* 遍历。

**示例:**

```
输入: [1,null,2,3]  
   1
    \
     2
    /
   3 

输出: [3,2,1]
```

**进阶:** 递归算法很简单，你可以通过迭代算法完成吗？

解答：

思路：

迭代

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        if not root:
            return res
        stack = [root]
        while stack:
            node = stack.pop()
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
            res.append(node.val)
        return res[::-1]
```

递归

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        if not root:
            return []
        if root.left:
            res += self.postorderTraversal(root.left)
        if root.right:
            res += self.postorderTraversal(root.right)
        res.append(root.val)
        return res
```



## 6. LRU缓存机制(Medium)

运用你所掌握的数据结构，设计和实现一个  [LRU (最近最少使用) 缓存机制](https://baike.baidu.com/item/LRU)。它应该支持以下操作： 获取数据 `get` 和 写入数据 `put` 。

获取数据 `get(key)` - 如果密钥 (key) 存在于缓存中，则获取密钥的值（总是正数），否则返回 -1。
写入数据 `put(key, value)` - 如果密钥不存在，则写入其数据值。当缓存容量达到上限时，它应该在写入新数据之前删除最近最少使用的数据值，从而为新的数据值留出空间。

**进阶:**

你是否可以在 **O(1)** 时间复杂度内完成这两种操作？

**示例:**

```
LRUCache cache = new LRUCache( 2 /* 缓存容量 */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // 返回  1
cache.put(3, 3);    // 该操作会使得密钥 2 作废
cache.get(2);       // 返回 -1 (未找到)
cache.put(4, 4);    // 该操作会使得密钥 1 作废
cache.get(1);       // 返回 -1 (未找到)
cache.get(3);       // 返回  3
cache.get(4);       // 返回  4
```

解答：

思路：

不用双向链表也可以，就是用字典保存key,value，用数组保存最近访问

```python
class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.stack = {}
        self.cache = []
        

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key in self.stack:
            self.cache.remove(key)
            self.cache.append(key)
            return self.stack[key]
        else:
            return -1
        
    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if key in self.stack:
            self.cache.remove(key)
        else:
            if len(self.stack) == self.capacity:
                del self.stack[self.cache[0]]
                self.cache.pop(0)            
        self.cache.append(key)
        self.stack[key] = value
        
# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```



## 7. 对链表进行插入排序(Medium)

对链表进行插入排序。

![img](https://upload.wikimedia.org/wikipedia/commons/0/0f/Insertion-sort-example-300px.gif)
插入排序的动画演示如上。从第一个元素开始，该链表可以被认为已经部分排序（用黑色表示）。
每次迭代时，从输入数据中移除一个元素（用红色表示），并原地将其插入到已排好序的链表中。

 

**插入排序算法：**

1. 插入排序是迭代的，每次只移动一个元素，直到所有元素可以形成一个有序的输出列表。
2. 每次迭代中，插入排序只从输入数据中移除一个待排序的元素，找到它在序列中适当的位置，并将其插入。
3. 重复直到所有输入数据插入完为止。

 

**示例 1：**

```
输入: 4->2->1->3
输出: 1->2->3->4
```

**示例 2：**

```
输入: -1->5->3->4->0
输出: -1->0->3->4->5
```

解答：

思路：插入排序

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return head
        dummy = ListNode(0)
        cur = head
        pre = dummy
        nex = None
        while cur:
            nex = cur.next
            while pre.next and pre.next.val < cur.val:
                pre = pre.next
            cur.next = pre.next
            pre.next = cur
            cur = nex
            pre = dummy
        return dummy.next
```



## 8. 排序链表(Medium)

在 *O*(*n* log *n*) 时间复杂度和常数级空间复杂度下，对链表进行排序。

**示例 1:**

```
输入: 4->2->1->3
输出: 1->2->3->4
```

**示例 2:**

```
输入: -1->5->3->4->0
输出: -1->0->3->4->5
```

解答：

思路：

归并排序的原理，很简单

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        slow,fast = head,head.next
        while fast and fast.next:
            fast,slow = fast.next.next,slow.next
        mid ,slow.next = slow.next,None
        left,right = self.sortList(head),self.sortList(mid)
        h = res = ListNode(0)
        while left and right:
            if left.val < right.val:
                h.next = left
                left = left.next
            else:
                h.next = right
                right = right.next
            h = h.next
        h.next = left if left else right
        return res.next
```



## 9. 直线上最多的点(Medium)

记录给定一个二维平面，平面上有 *n* 个点，求最多有多少个点在同一条直线上。

**示例 1:**

```
输入: [[1,1],[2,2],[3,3]]
输出: 3
解释:
^
|
|        o
|     o
|  o  
+------------->
0  1  2  3  4
```

**示例 2:**

```
输入: [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
输出: 4
解释:
^
|
|  o
|     o        o
|        o
|  o        o
+------------------->
0  1  2  3  4  5  6
```

解答：

思路：

这道题比较难

```python
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        def max_points_on_a_line(i):
            def add_line(i,j,count,dup):
                x1 = points[i][0]
                y1 = points[i][1]
                x2 = points[j][0]
                y2 = points[j][1]
                if x1 == x2 and y1 == y2:
                    dup += 1
                elif y1 == y2:
                    nonlocal hori
                    hori += 1
                    count = max(hori,count)
                else:
                    slope = (x1-x2)/(y1-y2)
                    lines[slope] = lines.get(slope,1) + 1
                    count = max(count,lines[slope])
                return count,dup
            lines = {}
            count = 1
            dup = 0
            hori = 1
            for j in range(i+1,n):
                count,dup = add_line(i,j,count,dup)
            return count + dup
        n = len(points)
        if n < 3:
            return n
        max_count = 1
        for i in range(n-1):
            max_count = max(max_count,max_points_on_a_line(i))
        return max_count
```



## 10. 逆波兰表达式求值(Medium)

根据[逆波兰表示法](https://baike.baidu.com/item/逆波兰式/128437)，求表达式的值。

有效的运算符包括 `+`, `-`, `*`, `/` 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。

**说明：**

- 整数除法只保留整数部分。
- 给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。

**示例 1：**

```
输入: ["2", "1", "+", "3", "*"]
输出: 9
解释: ((2 + 1) * 3) = 9
```

**示例 2：**

```
输入: ["4", "13", "5", "/", "+"]
输出: 6
解释: (4 + (13 / 5)) = 6
```

**示例 3：**

```
输入: ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
输出: 22
解释: 
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
```

## 11. 翻转字符串里的单词(Medium)

给定一个字符串，逐个翻转字符串中的每个单词。 

**示例 1：**

```
输入: "the sky is blue"
输出: "blue is sky the"
```

**示例 2：**

```
输入: "  hello world!  "
输出: "world! hello"
解释: 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
```

**示例 3：**

```
输入: "a good   example"
输出: "example good a"
解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
```

 

**说明：**

- 无空格字符构成一个单词。
- 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
- 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。

 

**进阶：**

请选用 C 语言的用户尝试使用 *O*(1) 额外空间复杂度的原地解法。

解答：

思路：

双指针做法

```python
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        s = s.strip()
        res = ''
        i,j = len(s)-1,len(s)
        while i > 0:
            if s[i] == ' ':
                res += s[i+1:j] + ' '
                while s[i] == ' ':
                    i -= 1
                j = i +1
            i -= 1
        return res + s[:j]
```



## 12. 乘积最大子序列(Medium)

给定一个整数数组 `nums` ，找出一个序列中乘积最大的连续子序列（该序列至少包含一个数）。

**示例 1:**

```
输入: [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
```

**示例 2:**

```
输入: [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
```

解答：

思路：

动态规划

保存一个局部最小值和最大值

遇到负数则交换两个值

```python
class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        minv,maxv,res = nums[0],nums[0],nums[0]
        for i in range(1,len(nums)):
            if nums[i]<0:
                minv,maxv = maxv,minv
            maxv = max(maxv*nums[i],nums[i])
            minv = min(minv*nums[i],nums[i])
            res = max(res,maxv)
        return res
```



## 13. 寻找旋转排序数组中的最小值(Medium)

假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 `[0,1,2,4,5,6,7]` 可能变为 `[4,5,6,7,0,1,2]` )。

请找出其中最小的元素。

你可以假设数组中不存在重复元素。

**示例 1:**

```
输入: [3,4,5,1,2]
输出: 1
```

**示例 2:**

```
输入: [4,5,6,7,0,1,2]
输出: 0
```

解答：

思路：

简单的二分法

```python
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l,r = 0,len(nums)-1
        while l < r:
            mid = (l+r)//2
            if nums[mid] > nums[r]:
                l = mid + 1
            else:
                r = mid
        return nums[l]
```



## 14. 寻找旋转排序数组中的最小值2(Hard)

假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 `[0,1,2,4,5,6,7]` 可能变为 `[4,5,6,7,0,1,2]` )。

请找出其中最小的元素。

注意数组中可能存在重复的元素。

**示例 1：**

```
输入: [1,3,5]
输出: 1
```

**示例 2：**

```
输入: [2,2,2,0,1]
输出: 0
```

**说明：**

- 这道题是 [寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/description/) 的延伸题目。
- 允许重复会影响算法的时间复杂度吗？会如何影响，为什么？

解答：

思路：

和上题一样，二分法

```python
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l,r = 0,len(nums)-1
        while l < r:
            mid = (l+r)//2
            if nums[mid] > nums[r]:
                l = mid + 1
            elif nums[mid] < nums[r]:
                r = mid
            else:
                r = r-1
        return nums[l]
```



## 15. 最小栈(Easy)

设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。

- push(x) -- 将元素 x 推入栈中。
- pop() -- 删除栈顶的元素。
- top() -- 获取栈顶元素。
- getMin() -- 检索栈中的最小元素。

**示例:**

```
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.
```

## 16. 上下翻转二叉树(Medium)

给定一个二叉树，其中所有的右节点要么是具有兄弟节点（拥有相同父节点的左节点）的叶节点，要么为空，将此二叉树上下翻转并将它变成一棵树， 原来的右节点将转换成左叶节点。返回新的根。

**例子:**

```
输入: [1,2,3,4,5]

    1
   / \
  2   3
 / \
4   5

输出: 返回二叉树的根 [4,5,2,#,#,3,1]

   4
  / \
 5   2
    / \
   3   1  
```

**说明:**

对 `[4,5,2,#,#,3,1]` 感到困惑? 下面详细介绍请查看 [二叉树是如何被序列化的](https://support.leetcode-cn.com/hc/kb/article/1194353/)。

二叉树的序列化遵循层次遍历规则，当没有节点存在时，'#' 表示路径终止符。

这里有一个例子:

```
   1
  / \
 2   3
    /
   4
    \
     5
```

上面的二叉树则被序列化为 `[1,2,3,#,#,4,#,#,5]`.

解答：

思路：

这道题很有意思，将左节点变成根节点，根节点变成右节点，右节点变成左节点

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def upsideDownBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root or not root.left:
            return root
        left = self.upsideDownBinaryTree(root.left)
        root.left.left = root.right
        root.left.right = root
        root.left = None
        root.right = None
        return left
```



## 17. 用Read4读取N个字符(Easy)

给你一个文件，并且该文件只能通过给定的 `read4` 方法来读取，请实现一个方法使其能够读取 n 个字符。

**read4 方法：**

API `read4` 可以从文件中读取 4 个连续的字符，并且将它们写入缓存数组 `buf` 中。

返回值为实际读取的字符个数。

注意 `read4()` 自身拥有文件指针，很类似于 C 语言中的 `FILE *fp` 。

**read4 的定义：**

```
参数类型: char[] buf
返回类型: int

注意: buf[] 是目标缓存区不是源缓存区，read4 的返回结果将会复制到 buf[] 当中。
```

下列是一些使用 `read4` 的例子：

```
File file("abcdefghijk"); // 文件名为 "abcdefghijk"， 初始文件指针 (fp) 指向 'a' 
char[] buf = new char[4]; // 创建一个缓存区使其能容纳足够的字符
read4(buf); // read4 返回 4。现在 buf = "abcd"，fp 指向 'e'
read4(buf); // read4 返回 4。现在 buf = "efgh"，fp 指向 'i'
read4(buf); // read4 返回 3。现在 buf = "ijk"，fp 指向文件末尾
```

**read 方法：**

通过使用 `read4` 方法，实现 `read` 方法。该方法可以从文件中读取 n 个字符并将其存储到缓存数组 `buf` 中。您 **不能** 直接操作文件。

返回值为实际读取的字符。

**read 的定义：**

```
参数类型:   char[] buf, int n
返回类型:   int

注意: buf[] 是目标缓存区不是源缓存区，你需要将结果写入 buf[] 中。
```

 

**示例 1：**

```
输入： file = "abc", n = 4
输出： 3
解释： 当执行你的 rand 方法后，buf 需要包含 "abc"。 文件一共 3 个字符，因此返回 3。 注意 "abc" 是文件的内容，不是 buf 的内容，buf 是你需要写入结果的目标缓存区。 
```

**示例 2：**

```
输入： file = "abcde", n = 5
输出： 5
解释： 当执行你的 rand 方法后，buf 需要包含 "abcde"。文件共 5 个字符，因此返回 5。
```

**示例 3:**

```
输入： file = "abcdABCD1234", n = 12
输出： 12
解释： 当执行你的 rand 方法后，buf 需要包含 "abcdABCD1234"。文件一共 12 个字符，因此返回 12。
```

**示例 4:**

```
输入： file = "leetcode", n = 5
输出： 5
解释： 当执行你的 rand 方法后，buf 需要包含 "leetc"。文件中一共 5 个字符，因此返回 5。
```

 

**注意：**

1. 你 **不能** 直接操作该文件，文件只能通过 `read4` 获取而 **不能** 通过 `read`。
2. `read`  函数只在每个测试用例调用一次。
3. 你可以假定目标缓存数组 `buf` 保证有足够的空间存下 n 个字符。 



## 18. 用Read4读取N个字符2(Hard)

给你一个文件，并且该文件只能通过给定的 `read4` 方法来读取，请实现一个方法使其能够读取 n 个字符。**注意：你的** **read 方法可能会被调用多次。**

**read4 的定义：**

```
参数类型: char[] buf
返回类型: int

注意: buf[] 是目标缓存区不是源缓存区，read4 的返回结果将会复制到 buf[] 当中。
```

下列是一些使用 `read4` 的例子：

```
File file("abcdefghijk"); // 文件名为 "abcdefghijk"， 初始文件指针 (fp) 指向 'a' 
char[] buf = new char[4]; // 创建一个缓存区使其能容纳足够的字符
read4(buf); // read4 返回 4。现在 buf = "abcd"，fp 指向 'e'
read4(buf); // read4 返回 4。现在 buf = "efgh"，fp 指向 'i'
read4(buf); // read4 返回 3。现在 buf = "ijk"，fp 指向文件末尾
```

**read 方法：**

通过使用 `read4` 方法，实现 `read` 方法。该方法可以从文件中读取 n 个字符并将其存储到缓存数组 `buf` 中。您 **不能** 直接操作文件。

返回值为实际读取的字符。

**read 的定义：**

```
参数:   char[] buf, int n
返回值: int

注意: buf[] 是目标缓存区不是源缓存区，你需要将结果写入 buf[] 中。
```

 

**示例 1：**

```
File file("abc");
Solution sol;
// 假定 buf 已经被分配了内存，并且有足够的空间来存储文件中的所有字符。
sol.read(buf, 1); // 当调用了您的 read 方法后，buf 需要包含 "a"。 一共读取 1 个字符，因此返回 1。
sol.read(buf, 2); // 现在 buf 需要包含 "bc"。一共读取 2 个字符，因此返回 2。
sol.read(buf, 1); // 由于已经到达了文件末尾，没有更多的字符可以读取，因此返回 0。
```

**Example 2:**

```
File file("abc");
Solution sol;
sol.read(buf, 4); // 当调用了您的 read 方法后，buf 需要包含 "abc"。 一共只能读取 3 个字符，因此返回 3。
sol.read(buf, 1); // 由于已经到达了文件末尾，没有更多的字符可以读取，因此返回 0。
```

**注意：**

1. 你 **不能** 直接操作该文件，文件只能通过 `read4` 获取而 **不能** 通过 `read`。
2. `read`  函数可以被调用 **多次**。
3. 请记得 **重置** 在 Solution 中声明的类变量（静态变量），因为类变量会 **在多个测试用例中保持不变**，影响判题准确。请 [查阅](https://support.leetcode-cn.com/hc/kb/section/1071534/) 这里。
4. 你可以假定目标缓存数组 `buf` 保证有足够的空间存下 n 个字符。 
5. 保证在一个给定测试用例中，`read` 函数使用的是同一个 `buf`。



## 19. 至多包含两个不同字符的最长子串(Hard)

给定一个字符串 **s** ，找出 **至多** 包含两个不同字符的最长子串 **t 。**

**示例 1:**

```
输入: "eceba"
输出: 3
解释: t 是 "ece"，长度为3。
```

**示例 2:**

```
输入: "ccaabbb"
输出: 5
解释: t 是 "aabbb"，长度为5。
```

解答：

思路：

滑动窗口，已经很熟练了，很简单的一道题

```python
class Solution(object):
    def lengthOfLongestSubstringTwoDistinct(self, s):
        """
        :type s: str
        :rtype: int
        """
        from collections import defaultdict
        lookup = defaultdict(int)
        start = end = 0
        max_len = 0
        cnt = 0
        while end < len(s):
            if lookup[s[end]] == 0:
                cnt += 1
            lookup[s[end]] += 1
            end +=1
            while cnt > 2:
                if lookup[s[start]] == 1:
                    cnt -= 1
                lookup[s[start]] -= 1
                start += 1
            max_len = max(max_len,end-start)
        return max_len
```



## 20. 相交链表(Easy)

编写一个程序，找到两个单链表相交的起始节点。

如下面的两个链表**：**

[![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png)](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png)

在节点 c1 开始相交。

 

**示例 1：**

[![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_1.png)](https://assets.leetcode.com/uploads/2018/12/13/160_example_1.png)

```
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Reference of the node with value = 8
输入解释：相交节点的值为 8 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
```

 

**示例 2：**

[![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_2.png)](https://assets.leetcode.com/uploads/2018/12/13/160_example_2.png)

```
输入：intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Reference of the node with value = 2
输入解释：相交节点的值为 2 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [0,9,1,2,4]，链表 B 为 [3,2,4]。在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。
```

 

**示例 3：**

[![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_3.png)](https://assets.leetcode.com/uploads/2018/12/13/160_example_3.png)

```
输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：null
输入解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
解释：这两个链表不相交，因此返回 null。
```

 

**注意：**

- 如果两个链表没有交点，返回 `null`.
- 在返回结果后，两个链表仍须保持原有的结构。
- 可假定整个链表结构中没有循环。
- 程序尽量满足 O(*n*) 时间复杂度，且仅用 O(*1*) 内存。

解答：

思路：双指针加链表拼接

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        ha,hb = headA,headB
        while ha != hb:
            ha = ha.next if ha else headB
            hb = hb.next if hb else headA
        return ha
```

