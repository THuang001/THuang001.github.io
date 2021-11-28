---
title: leetcode81-85
date: 2019-06-26 20:46:43
tags: leetcode
categories: 算法
---

## 1. 搜索旋转排序数组2(Medium)

假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 `[0,0,1,2,2,5,6]` 可能变为 `[2,5,6,0,0,1,2]` )。

编写一个函数来判断给定的目标值是否存在于数组中。若存在返回 `true`，否则返回 `false`。

**示例 1:**

```
输入: nums = [2,5,6,0,0,1,2], target = 0
输出: true
```

**示例 2:**

```
输入: nums = [2,5,6,0,0,1,2], target = 3
输出: false
```

**进阶:**

- 这是 [搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/description/) 的延伸题目，本题中的 `nums`  可能包含重复元素。
- 这会影响到程序的时间复杂度吗？会有怎样的影响，为什么？

**解答：**

**思路：二分法**

判断二分点,几种可能性

1. 直接nums[mid] == target

2. 当数组为[1,2,1,1,1],nums[mid] == nums[left] == nums[right],需要left++, right --;

3. 当nums[left]<= nums[mid],说明是在左半边的递增区域

	a. nums[left] <=target < nums[mid],说明target在left和mid之间.我们令right = mid - 1;
	
	b. 不在之间, 我们令 left = mid + 1;

4. 当nums[mid] < nums[right],说明是在右半边的递增区域

	a. nums[mid] < target <= nums[right],说明target在mid 和right之间,我们令left = mid + 1
	
	b. 不在之间,我们令right = mid - 1;

时间复杂度:$O(logn)$

```python
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        l,r = 0,len(nums)-1
        while l <= r:
            mid = (l+r)>>1
            if nums[mid] == target:
                return True
            if nums[mid] == nums[l] == nums[r]:
                l += 1
                r -= 1
            elif nums[mid] >= nums[l]:
                if nums[l] <= target < nums[mid]:
                    r = mid-1
                else:
                    l = mid+1
            else:
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid-1
        return False
```



## 2. 删除排序链表中的重复项2(Medium)

给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中 *没有重复出现* 的数字。

**示例 1:**

```
输入: 1->2->3->3->4->4->5
输出: 1->2->5
```

**示例 2:**

```
输入: 1->1->1->2->3
输出: 2->3
```

**解答：**

**思路：双指针大法，快慢指针**

用快指针跳过那些有重复数组,慢指针负责和快指针拼接!

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        while not head or not head.next:
            return head
        dummy = ListNode(-1)
        dummy.next = head
        slow = dummy
        fast = dummy.next
        while fast:
            if fast.next and fast.val == fast.next.val:
                tmp = fast.val
                while fast and fast.val == tmp:
                    fast = fast.next
            else:
                slow.next = fast
                slow = fast
                fast = fast.next
        slow.next = fast
        return dummy.next
```

**执行用时 :28 ms, 在所有 Python 提交中击败了96.83%**

**思路二：更省时间的方法是用一个prev 和 cur 指针，这样loop一次即可解决问题。**

pre描述重复字段的左起点

head不断遍历达到重复字段的右端点，同时遍历得到新的重复字段的起点。

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        while not head or not head.next:
            return head
        dummy = cur = pre = ListNode(None)
        while head:
            while head and ((head.val == pre.val) or (head.next and head.val == head.next.val)):
                pre = head
                head = head.next
            cur.next = head
            cur = cur.next
            if head:
                head = head.next
        return dummy.next
```



## 3. 删除排序链表中的重复项(Easy)

给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。

**示例 1:**

```
输入: 1->1->2
输出: 1->2
```

**示例 2:**

```
输入: 1->1->2->3->3
输出: 1->2->3
```

**解答：**

**思路：dummy大法**

很简单啊，dummy大法在解决这种链表问题上，很有用。

```python
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = head
        while head:
            while head.next and head.next.val == head.val:
                head.next = head.next.next    # skip duplicated node
            head = head.next     # not duplicate of current node, move to next node
        return dummy  
```



## 4. 柱状图中最大的矩形(Medium)

给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

![](images/histogram.png) 



以上是柱状图的示例，其中每个柱子的宽度为 1，给定的高度为 `[2,1,5,6,2,3]`。

![](../images/histogram_area.png)



图中阴影部分为所能勾勒出的最大矩形面积，其面积为 `10` 个单位。

 

**示例:**

```
输入: [2,1,5,6,2,3]
输出: 10
```

**解答：**

**思路：**

暴力解法$O(n^2)$

超时，很简单。

首先，要想找到第 i 位置最大面积是什么？

是以i 为中心，向左找第一个小于 heights[i] 的位置 left_i；向右找第一个小于于 heights[i] 的位置 right_i，即最大面积为 heights[i] * (right_i - left_i -1)

```python
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        res = 0
        n = len(heights)
        for i in range(n):
            l = i
            r = i
            while l >= 0 and heights[l] >= heights[i]:
                l -= 1
            while r < n and heights[r] >= heights[i]:
                r += 1
            res = max(res,heights[i]*(r-l-1))
        return res
```

**暴力解法2:思路很简单，同样也超时**

```python
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        res = 0
        n = len(heights)
        for i in range(n):
            minHeight = float('inf')
            for j in range(i,n):
                minHeight = min(minHeight,heights[j])
                res = max(res,minHeight*(j-i+1))
        return res
```

思路三：这个题是真的难。。。

```python
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        stack = [-1]
        res = 0
        n = len(heights)
        for i in range(n):
            while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
                tmp = stack.pop()
                res = max(res,heights[tmp] * (i-stack[-1]-1))
            stack.append(i)
        while stack[-1] != -1:
            tmp = stack.pop()
            res = max(res,heights[tmp]*(n-stack[-1]-1))
        return res
```

我个人觉得，下面这个解答，更容易理解。

```python
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        heights.append(0)
        area, stack = 0, [-1]
        for idx, height in enumerate(heights):
            while height < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = idx - stack[-1] - 1
                area = max(w*h, area)
            stack.append(idx)
        return area
```



## 5. 最大矩形(Hard)

给定一个仅包含 0 和 1 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。

**示例:**

```
输入:
[
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
输出: 6
```

**解答：**

**思路：**

巧用上一题的思路，分行读取，每一行高度累积到一个长度为col的数组中，

将问题转化为上一题的意思。

每一行都执行一遍84题



```python
class Solution(object):
    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if not matrix or not matrix[0]:
            return 0
        row = len(matrix)
        col = len(matrix[0])
        height = [0] * (col+2)
        res = 0
        for i in range(row):
            stack = []
            for j in range(col+2):
                if 1<= j<=col:
                    if matrix[i][j-1] == '1':
                        height[j] += 1
                    else:
                        height[j] = 0
                while stack and height[stack[-1]] > height[j]:
                    cur = stack.pop()
                    res = max(res,(j-stack[-1]-1)*height[cur])
                stack.append(j)
        return res
```

**思路二：分行读取，然后将每一列的高度加一存到数组中，每一列逐步更新得到该列中小于该高度的最小左边界和最小右边界。然后计算面积。**

```python
class Solution(object):
    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if not matrix or not matrix[0]:
            return 0
        row = len(matrix)
        col = len(matrix[0])
        left = [-1]*col
        right = [col]*col
        height = [0]*col
        res = 0
        for i in range(row):
            cur_l = -1
            cur_r = col
            for j in range(col):
                if matrix[i][j] == '1':
                    height[j] += 1
                else:
                    height[j] = 0
                    
            for j in range(col):
                if matrix[i][j] == '1':
                    left[j] = max(left[j],cur_l)
                else:
                    left[j] = -1
                    cur_l = j
            
            for j in range(col-1,-1,-1):
                if matrix[i][j] == '1':
                    right[j] = min(right[j],cur_r)
                else:
                    right[j] = col
                    cur_r = j
            
            for j in range(col):
                res = max(res,(right[j] - left[j] - 1)*height[j])
        return res
```

