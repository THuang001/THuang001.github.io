---
title: leetcode1-3
date: 2019-06-17 17:31:56
tags: leetcode
categories: 算法
---





## 1. 两数之和(Easy)

给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

**示例：**

    给定 nums = [2, 7, 11, 15], target = 9
    因为 nums[0] + nums[1] = 2 + 7 = 9，所以返回 [0, 1]

**解答：**

***思路一：时间复杂度$O(n^2)$，空间复杂度：O(1)***

暴力解法，两轮遍历

第一轮取出`index`为`i`的数`num1`，第二轮取出更靠后的`index`为`j`的`num2`

- 如果第二轮取出的是`num1`之前的数，其实我们之前已经考虑到这种情况了
- 如果第二轮再取`num1`的话，就不符合题目要求了

题目要求只需要找到一种，所以一旦找到就返回。

时间复杂度中的`N`是代表`nums`序列的长度。

```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                if nums[i] + nums[j] == target:
                    return [i,j]
```



***思路二：时间复杂度$O(n)$，空间复杂度$O(n)$***

上面思路一的时间复杂度太高了，典型的加快时间的方法有牺牲空间换取时间。

我们希望在我们的顺序遍历中取得一个数`num1`的时候，就知道和它配对的数是否在我们的`nums`中，并且不单单是存在，比如说`target`为`4`，`nums` 为`[2,3]`，假设我们此时取得的`num1`为`2`，那么和它配对的`2`确实在`nums`中，但是数字`2`在`nums`中只出现了一次，我们无法取得两次，所以也是不行的。

因此我们有了下面的步骤

1. 建立字典 `lookup` 存放第一个数字，并存放该数字的 `index`
2. 判断 `lookup` 中是否存在 `target - 当前数字cur`， 则当前值`cur`和某个`lookup`中的`key`值相加之和为 `target`.
3. 如果存在，则返回： `target - 当前数字cur` 的 `index` 与 当前值`cur`的`index`

```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        lookup = {}
        for i,num in enumerate(nums):
            if target - num in lookup:
                return [lookup[target-num],i]
            else:
                lookup[num] = i
```



就像之前提到的特殊情况一样，这里注意我们要**边遍历边将 `num: idx`放入`lookup`中**，而不是在做遍历操作之前就将所有内容放入`lookup`中。



## 2. 两数相加(Easy)

给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。

如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。

您可以假设除了数字 0 之外，这两个数都不会以 0 开头。

**示例：**

    输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
    输出：7 -> 0 -> 8
    原因：342 + 465 = 807

**解答：**

***思路一：时间复杂度$O(n)$，空间复杂度$O(n)$***

将`l1`和`l2`全部变成数字做加法再换回去呗，这是我们最直接的想法。



```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        val1, val2 = [l1.val], [l2.val]

        while l1.next:
            val1.append(l1.next.val)
            l1 = l1.next
        while l2.next:
            val2.append(l2.next.val)
            l2 = l2.next

        # 求出 l1 和 l2 代表的数字
        num1 = ''.join([str(i) for i in val1[::-1]])
        num2 = ''.join([str(i) for i in val2[::-1]])

        # 得到 l1 和 l2 相加之和
        sums = int(num1) + int(num2)

        # 将 sums 转成题目中 linkedlist 所对应的表示形式
        sums = str(sums)[::-1]

        # dummy 作为返回结果
        dummy = head = ListNode(int(sums[0]))
        for i in range(1, len(sums)):
            head.next = ListNode(int(sums[i]))
            head = head.next
        return dummy
```

***思路二：时间复杂度$O(n)$，空间复杂度$O(1)$***

因为时间复杂度无法减小，我们一定得遍历完`l1`和`l2`的每一位才能得到最终的结果，`O(N)`没得商量

但是我们可以考虑减小我们的空间复杂度了，刚才我们是将`l1`和`l2`全部转回数字，然后用两个列表将他们的数字形式存了下来，这消耗了`O(N)`的空间。

实际上我们完全可以模拟真正的加法操作，即从个位数开始相加，如果有进位就记录一下，等到十位数相加的时候记得加上那个进位`1`就可以了，这是我们小学就学过的知识。

那么我们就先处理个位数的相加，然后我们发现处理十位数，百位数和后面的位数都和个位数相加的操作是一个样子的，只不过后面计算的结果乘上`10`再加上个位数相加的结果，这才是最终的结果。

于是我们就想到了用递归的方法，即一步一步将大问题转化为更小的问题，直到遇到基础情况（这里指的是个位数相加）返回即可。

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if not l1 and not l2:
            return
        elif not (l1 and l2):
            return l1 or l2
        else:
            if l1.val + l2.val < 10:
                l3 = ListNode(l1.val + l2.val)
                l3.next = self.addTwoNumbers(l1.next,l2.next)
            else:
                l3 = ListNode(l1.val+l2.val-10)
                l3.next = self.addTwoNumbers(l1.next,self.addTwoNumbers(l2.next,ListNode(1)))
        return l3
```



## 3. 无重复字符的最长字串(Medium)

给定一个字符串，请你找出其中不含有重复字符的 **最长子串** 的长度。

**示例 1:**

```
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

**示例 2:**

```
输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
```

**示例 3:**

```
输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```



***思路一：时间复杂度$O(n)$，空间复杂度$O(n)$***

求一个最长的字串，里面不带任何重复字符。

假设`input`为`"abcabcbb"`，我们先从第一个字符开始，只有一个字符肯定不会重复吧，`“a”`满足条件，更新最大长度为`1`，然后走到第二个字符，`“ab”`也满足，更新最大长度为`2`，走到第三个字符，`“abc”`也满足，更新最大长度为`3`，走到第四个字符，我们发现`“a”`已经出现过了，于是我们就必须要删除之前的一些字符来继续满足无重复字符的条件，但是我们不知道前面已经出现过一次的`“a”`的`index`在哪里呀，所以我们只能一个一个找了，从当前子串的`“abca”`的第一个字符开始找，删除第一个字符`“a”`，发现这时候只剩下一个`“a”`了，我们又满足条件了，更新最大长度为`3`，以此类推

```
start
end
 | 
 | 
 v 

 a  b  c  a  b  c  b  b 

end 指针不停往前走，只要当前子串 s[start:end+1] 不满足无重复字符条件的时候，我们就让 start 指针往前走直到满足条件为止，每次满足条件我们都要update一下最大长度，即 res
```

**滑动窗口slide window**

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        
        start,end = 0,0
        res,lookup = 0,set()
        while start < len(s) and end < len(s):
            if s[end] not in lookup:
                lookup.add(s[end])
                res = max(res,end-start+1)
                end += 1
            else:
                lookup.discard(s[start])
                start += 1
                
        return res
```

***思路二：时间复杂度时间复杂度$O(n)$，空间复杂度$O(n)$***

那么为了之后 `LeetCode` 里面一些类似的题目，我们这里做一个 `slide window` 的模版，以后就可以重复使用了

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        lookup = collections.defaultdict(int)
        l,r,counter,res = 0,0,0,0
        while r < len(s):
            lookup[s[r]] += 1
            if lookup[s[r]] == 1:
                counter += 1
            r += 1
            
            while l < r and counter < r - l:
                lookup[s[l]] -= 1
                if lookup[s[l]] == 0:
                    counter -= 1
                
                l += 1
            res = max(res,r - l)
        return res
```



***思路三：时间复杂度时间复杂度$O(n)$，空间复杂度$O(n)$***

刚才思路一中有这样一句话：`但是我们不知道前面已经出现过一次的“a”的index在哪里呀，所以我们只能一个一个找了`

我们可以对这里做一个优化，就不需要一个个去找了，我们只需要用一个字典，对于当前子串中的每一个字符，将其在`input`中的来源`index`记录下来即可

我们先从第一个字符开始，只要碰到已经出现过的字符我们就必须从之前出现该字符的`index`开始重新往后看。

例如`‘xyzxlkjh’`，当看到第二个`‘x’`时我们就应该从第一个`x`后面的`y`开始重新往后看了。

我们将每一个已经阅读过的字符作为`key`，而它的值就是它在原字符串中的`index`，如果我们现在的字符不在字典里面我们就把它加进字典中去，因此，只要`end`指针指向的的这个字符`c`在该字典中的值大于等于了当前子串首字符的`index`时，就说明`c`在当前子串中已经出现过了，我们就将当前子串的首字符的`index`加`1`，即从后一位又重新开始读，此时当前子串已经满足条件了，然后我们更新`res`。

#### 程序变量解释

- `start` 是当前无重复字符的子串首字符的 `index`
- `maps` 放置每一个字符的 `index`，如果 `maps.get(s[i], -1)` 大于等于 `start` 的话，就说明字符重复了，此时就要重置 `res` 和 `start` 的值了

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        maps = {}
        res,start = 0,0
        for i in range(len(s)):
            start = max(start,maps.get(s[i],-1) + 1)
            res = max(res,i - start + 1)
            maps[s[i]] = i
        return res
```

