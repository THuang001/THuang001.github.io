---
title: leetcode26-30
date: 2019-06-20 22:54:43
tags: leetcode
categories: 算法
---

## 1. 删除排序数组中的重复项(Easy)

给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。

不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。

**示例 1:**

```
给定数组 nums = [1,1,2], 

函数应该返回新的长度 2, 并且原数组 nums 的前两个元素被修改为 1, 2。 

你不需要考虑数组中超出新长度后面的元素。
```

**示例 2:**

```
给定 nums = [0,0,1,1,1,2,2,3,3,4],

函数应该返回新的长度 5, 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。

你不需要考虑数组中超出新长度后面的元素。
```

**说明:**

为什么返回数值是整数，但输出的答案是数组呢?

请注意，输入数组是以“引用”方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。

你可以想象内部操作如下:

```
// nums 是以“引用”方式传递的。也就是说，不对实参做任何拷贝
int len = removeDuplicates(nums);

// 在函数里修改输入数组对于调用者是可见的。
// 根据你的函数返回的长度, 它会打印出数组中该长度范围内的所有元素。
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
```

**解答：**

**思路一：双指针法**

**快指针遍历旧数组，满指针指向新数组**

```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        slow = 0
        for fast in range(len(nums)):
            if nums[slow] != nums[fast]:
                slow += 1
                nums[slow] = nums[fast]
        return slow+1
```

思路二：

很简单，一看就明白

```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        idx = 0
        for num in nums:
            if idx < 1 or num != nums[idx-1]:
                nums[idx] = num
                idx += 1
        return idx
```



## 2. 移除元素(Easy)

给定一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素，返回移除后数组的新长度。

不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

**示例 1:**

```
给定 nums = [3,2,2,3], val = 3,

函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。

你不需要考虑数组中超出新长度后面的元素。
```

**示例 2:**

```
给定 nums = [0,1,2,2,3,0,4,2], val = 2,

函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。

注意这五个元素可为任意顺序。

你不需要考虑数组中超出新长度后面的元素。
```

**说明:**

为什么返回数值是整数，但输出的答案是数组呢?

请注意，输入数组是以“引用”方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。

你可以想象内部操作如下:

```
// nums 是以“引用”方式传递的。也就是说，不对实参作任何拷贝
int len = removeElement(nums, val);

// 在函数里修改输入数组对于调用者是可见的。
// 根据你的函数返回的长度, 它会打印出数组中该长度范围内的所有元素。
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
```

解答：

**思路：**

**这个题很简单。思路也很简单。**

如果当前数字等于$val$，就把当前数字换成数组的最后一个数字，然后删除掉数组最后一个数字，这样可以实现时间$O(N)$了

```python
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        idx = 0
        while idx < len(nums):
            if nums[idx] == val:
                nums[idx] = nums[-1]
                del nums[-1]
            else:
                idx += 1
        return len(nums)
```

**思路二：双指针法**

**双指针法真是好用，在数组问题里面。**

```python
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
        return slow        
```



## 3. 实现strStr()(Easy)

实现 strStr() 函数。

给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。

**示例 1:**

```
输入: haystack = "hello", needle = "ll"
输出: 2
```


**示例 2:**

```
输入: haystack = "aaaaa", needle = "bba"
输出: -1
```


**说明:**

当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。

对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与C语言的 strstr() 以及 Java的 indexOf() 定义相符。

**解答：**

**思路一：**

**KMP算法**

```python
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """        
        if not needle : return 0
        _next = [0] * len(needle)
        def getNext(p, _next):
            _next[0] = -1
            i = 0
            j = -1
            while i < len(p) - 1:
                if j == -1 or p[i] == p[j]:
                    i += 1
                    j += 1
                    _next[i] = j
                else:
                    j = _next[j]
        getNext(needle, _next)
        i = 0
        j = 0
        while i < len(haystack) and j < len(needle):
            if j == -1 or haystack[i] == needle[j]:
                i += 1
                j += 1
            else:
                j = _next[j]
        if j == len(needle):
            return i - j
        return -1
```

**思路二：暴力搜索**

```python
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """    
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i:i+len(needle)] == needle:
                return i
        return -1
```

## 4. 两数相除(Medium)

给定两个整数，被除数 dividend 和除数 divisor。将两数相除，要求不使用乘法、除法和 mod 运算符。

返回被除数 dividend 除以除数 divisor 得到的商。

**示例 1:**

```
输入: dividend = 10, divisor = 3
输出: 3
```

**示例 2:**

```
输入: dividend = 7, divisor = -3
输出: -2
```

**说明:**

1. 被除数和除数均为 32 位有符号整数。
2. 除数不为 0。

假设我们的环境只能存储 32 位有符号整数，其数值范围是 $[−2^{31},  2^{31} − 1]$。本题中，如果除法结果溢出，则返回 $2^{31} − 1$。

**解答：**

**思路一：**

不能乘除，所以我们改用位运算，运用二分的思想，beats 100%

因为最坏的情况就是最大的数除以1，即`2**31 / 1`, 所以时间复杂度就是O(1)，如果assuming n is the number of bits， 那就是O(lgN)

```python
class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        res = 0
        positive = 1 if dividend^divisor>=0 else -1
        dividend = abs(dividend)
        divisor = abs(divisor)
        while dividend >= divisor:
            tmp,cnt = divisor,1
            while dividend >= tmp:
                dividend -= tmp
                res += cnt
                tmp <<= 1
                cnt <<= 1
        res = positive * res
        return max(min(res,2**31-1),-2**31)
```



## 5. 串联所有单词的字串(Hard)

给定一个字符串 s 和一些长度相同的单词 words。找出 s 中恰好可以由 words 中所有单词串联形成的子串的起始位置。

注意子串要与 words 中的单词完全匹配，中间不能有其他字符，但不需要考虑 words 中单词串联的顺序。 

**示例 1：**

```
输入：
  s = "barfoothefoobarman",
  words = ["foo","bar"]
输出：[0,9]
解释：
从索引 0 和 9 开始的子串分别是 "barfoor" 和 "foobar" 。
输出的顺序不重要, [9,0] 也是有效答案。
```


**示例 2：**

```
输入：
  s = "wordgoodgoodgoodbestword",
  words = ["word","good","best","word"]
输出：[]
```

**解答：**

**思路：滑动窗口+双指针**

把words里面的word和其频率都记下来，然后对s做双指针记录，只要匹配上了words就append到res中去

但是由于从不同起点开始结果不一样，所以最前面还要做一个for循环`for i in range(word_len):`这样从每个起点开始都能到达最后

```python
class Solution(object):
    def findSubstring(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: List[int]
        """
        from collections import Counter
        if not s or not words:
            return []
        one_word = len(words[0])
        word_num = len(words)
        n = len(s)
        if n < one_word:
            return []
        words = Counter(words)
        res = []
        for i in range(0,one_word):
            cur_cnt = 0
            left = i
            right = i
            cur_Counter = Counter()
            while right+one_word<=n:
                w = s[right:right+one_word]
                right += one_word
                if w not in words:
                    left = right
                    cur_Counter.clear()
                    cur_cnt = 0
                else:
                    cur_Counter[w] += 1
                    cur_cnt += 1
                    while cur_Counter[w] > words[w]:
                        left_w = s[left:left+one_word]
                        left += one_word
                        cur_Counter[left_w] -= 1
                        cur_cnt -= 1
                    if cur_cnt == word_num:
                        res.append(left)
        return res
```

