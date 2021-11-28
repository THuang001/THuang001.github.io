---
title: 滑动窗口类题目(本章最后一题有惊喜)
date: 2019-07-10 15:20:14
tags: sliding window
categories: 算法
---

# 一、综述

这类题在leetcode上，绝大部分都是难题，然后核心在于双指针技巧。

本文讲的例题从第3题覆盖到第727题。

# 二、滑动窗口思路讲解

在滑动窗口类型的问题中都会有两个指针。**一个用于延伸现有窗口的 right指针，和一个用于收缩窗口的left 指针。在任意时刻，只有一个指针运动，而另一个保持静止。**

本题的解法很符合直觉。我们通过移动right指针不断扩张窗口。当窗口包含全部所需的字符后，如果能收缩，我们就收缩窗口直到得到最小窗口。

答案是最小的可行窗口。





滑动窗口算法的思路是这样：

1、我们在字符串 S 中使用双指针中的左右指针技巧，初始化 left = right = 0，把索引闭区间 [left, right] 称为一个「窗口」。

2、我们先不断地增加 right 指针扩大窗口 [left, right]，直到窗口中的字符串符合要求（包含了 T 中的所有字符）。

3、此时，我们停止增加 right，转而不断增加 left 指针缩小窗口 [left, right]，直到窗口中的字符串不再符合要求（不包含 T 中的所有字符了）。同时，每次增加 left，我们都要更新一轮结果。

4、重复第 2 和第 3 步，直到 right 到达字符串 S 的尽头。

这个思路其实也不难，第 2 步相当于在寻找一个「可行解」，然后第 3 步在优化这个「可行解」，最终找到最优解。左右指针轮流前进，窗口大小增增减减，窗口不断向右滑动。

滑动窗口的抽象算法思想：

```c++
int left = 0, right = 0;

while (right < s.size()) {
    window.add(s[right]);
    right++;
    
    while (valid) {
        window.remove(s[left]);
        left++;
    }
}
```

使用滑动窗口解题的主要思想详细模板：

```python
        if not s:
            return 0
        from collections import defaultdict
        lookup = defaultdict(int)
        start,end = 0,0
        max_len,counter = 0,0
        while end < len(s):
            if lookup(s[end]) > 0:
                counter += 1
            lookup[s[end]] += 1
            end += 1
            while counter > 0:
                if lookup[s[start]] > 1:
                    counter -= 1
                lookup[s[start]] -= 1
                start += 1
            max_len = max(max_len,end-start)
        return max_len
```



# 三、例题讲解

## 1. 无重复字符的最长子串

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

### 思路：

这道题主要用到思路是：滑动窗口

什么是滑动窗口？

其实就是一个队列,比如例题中的 abcabcbb，进入这个队列（窗口）为 abc 满足题目要求，当再进入 a，队列变成了 abca，这时候不满足要求。所以，我们要移动这个队列！

如何移动？

我们只要把队列的左边的元素移出就行了，直到满足题目要求！

一直维持这样的队列，找出队列出现最长的长度时候，求出解！

时间复杂度：O(n) 

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        left = 0
        lookup = set()
        n = len(s)
        max_len,cur_len = 0,0
        for i in range(n):
            cur_len += 1
            while s[i] in lookup:
                lookup.remove(s[left])
                left += 1
                cur_len -= 1
            max_len = max(max_len,cur_len)
            lookup.add(s[i])
        return max_len
```

## 2. 串联所有单词的子串

给定一个字符串 **s** 和一些长度相同的单词 **words。**找出 **s** 中恰好可以由 **words** 中所有单词串联形成的子串的起始位置。

注意子串要与 **words** 中的单词完全匹配，中间不能有其他字符，但不需要考虑 **words** 中单词串联的顺序。

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

### 思路：

这个就很简单了，我们可以想象成一直在维护一个窗口，右指针每次移动one_word长度，看是否满足题意了，如果不满足，则继续寻找，如果出现新词，则从左边开始删除，直到没有新词。



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
        words = Counter(words)
        res = []
        for i in range(0,one_word):
            cur_cnt = 0
            left = right = i
            cur_Counter = Counter()
            while right+one_word <= n:
                w = s[right:right + one_word]
                right += one_word
                cur_cnt += 1
                cur_Counter[w] += 1
                while cur_Counter[w] > words[w]:
                    left_w = s[left:left+one_word]
                    left += one_word
                    cur_Counter[left_w] -= 1
                    cur_cnt -= 1
                if cur_cnt == word_num:
                    res.append(left)
        return res
```

## 3. 最小覆盖子串

给你一个字符串 S、一个字符串 T，请在字符串 S 里面找出：包含 T 所有字母的最小子串。

**示例：**

```
输入: S = "ADOBECODEBANC", T = "ABC"
输出: "BANC"
```

**说明：**

- 如果 S 中不存这样的子串，则返回空字符串 `""`。
- 如果 S 中存在这样的子串，我们保证它是唯一的答案。

### 思路：

保存一个滑动窗口，end<len(s)来移动，count==0来判断是否覆盖完全，如果覆盖了，则从左边开始删除窗口数据，调整窗口大小，同时更新。

```python
class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        from collections import defaultdict
        lookup = defaultdict(int)
        for c in t:
            lookup[c] += 1
        start = end = 0
        min_len = float('inf')
        count = len(t)
        res = ''
        while end < len(s):
            if lookup[s[end]] > 0:
                count -= 1
            lookup[s[end]] -= 1
            end += 1
            while count == 0:
                if min_len>end - start:
                    min_len = end-start
                    res = s[start:end]
                if lookup[s[start]] == 0:
                    count += 1
                lookup[s[start]] += 1
                start += 1
        return res
```

## 4. 至多包含两个不同字符的最长子串

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

### 思路：

这道题思路就是模板思路，end<len(s)来右移动指针，cnt>2时来左移动指针。

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





## 5. 至多包含K个不同字符的最长子串

记录

给定一个字符串 **s** ，找出 **至多** 包含 *k* 个不同字符的最长子串 **T**。

**示例 1:**

```
输入: s = "eceba", k = 2
输出: 3
解释: 则 T 为 "ece"，所以长度为 3。
```

**示例 2:**

```
输入: s = "aa", k = 1
输出: 2
解释: 则 T 为 "aa"，所以长度为 2。
```

### 思路：

怎么说，这道题和上面一道题就是同一道题。。。用滑动窗口来做，就改了一个参数。

```python
class Solution(object):
    def lengthOfLongestSubstringKDistinct(self, s, k):
        """
        :type s: str
        :type k: int
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
            while cnt > k:
                if lookup[s[start]] == 1:
                    cnt -= 1
                lookup[s[start]] -= 1
                start += 1
            max_len = max(max_len,end-start)
        return max_len
```

## 6. 滑动窗口最大值

给定一个数组 *nums*，有一个大小为 *k* 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口 *k* 内的数字。滑动窗口每次只向右移动一位。

返回滑动窗口最大值。

**示例:**

```
输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
输出: [3,3,5,5,6,7] 
解释: 

  滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

**注意：**

你可以假设 *k* 总是有效的，1 ≤ k ≤ 输入数组的大小，且输入数组不为空。

**进阶：**

你能在线性时间复杂度内解决此题吗？

### 思路：

也是滑动窗口。。

```python
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        start,end = 0,0
        cnt = 0
        res,tmp = [],[]
        while end < len(nums):
            cnt += 1
            tmp.append(nums[end])
            end += 1
            while cnt > k:
                tmp.remove(nums[start])
                start += 1
                cnt -= 1
            if cnt == k:
                res.append(max(tmp))
        return res
```

## 7. 长度最小的子数组

给定一个含有 **n** 个正整数的数组和一个正整数 **s ，**找出该数组中满足其和 **≥ s** 的长度最小的连续子数组**。**如果不存在符合条件的连续子数组，返回 0。

**示例:** 

```
输入: s = 7, nums = [2,3,1,2,4,3]
输出: 2
解释: 子数组 [4,3] 是该条件下的长度最小的连续子数组。
```

**进阶:**

如果你已经完成了*O*(*n*) 时间复杂度的解法, 请尝试 *O*(*n* log *n*) 时间复杂度的解法。

### 思路：

就是标准的滑动窗口解法。

```python
class Solution(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        if not nums or len(nums) == 0:
            return 0
        start,end = 0,0
        tmp_sum = 0
        res = float('inf')
        while end < len(nums):
            tmp_sum += nums[end]
            end += 1
            while tmp_sum >= s:
                res = min(res,end-start)
                tmp_sum -= nums[start]
                start += 1
        return 0 if res == float('inf') else res
```

## 8. 最小窗口子序列

给定字符串 `S` and `T`，找出 `S` 中最短的（连续）**子串** `W` ，使得 `T` 是 `W` 的 **子序列**。

如果 `S` 中没有窗口可以包含 `T` 中的所有字符，返回空字符串 `""`。如果有不止一个最短长度的窗口，返回开始位置最靠左的那个。

**示例 1：**

```
输入：
S = "abcdebdde", T = "bde"
输出："bcde"
解释：
"bcde" 是答案，因为它在相同长度的字符串 "bdde" 出现之前。
"deb" 不是一个更短的答案，因为在窗口中必须按顺序出现 T 中的元素。
```

 

**注：**

- 所有输入的字符串都只包含小写字母。All the strings in the input will only contain lowercase letters.
- `S` 长度的范围为 `[1, 20000]`。
- `T` 长度的范围为 `[1, 100]`。

### 思路：

哈哈哈这个题尝试用滑动窗口不能解决，因为这是序列，不是子串，要考虑字符的前后顺序关系。所以下面的代码没有通过。得到的结果是'deb'

```python
class Solution(object):
    def minWindow(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: str
        """
        from collections import defaultdict
        lookup = defaultdict(int)
        for c in T:
            lookup[c] += 1
        start,end = 0,0
        count = len(T)
        min_len = float('inf')
        res = ''
        while end < len(S):
            if lookup[S[end]] > 0:
                count -= 1
            lookup[S[end]] -= 1
            end += 1
            while count == 0:
                if min_len > end - start:
                    min_len = end-start
                    res = S[start:end]
                if lookup[S[start]] == 0:
                    count += 1
                lookup[S[start]] += 1
                start += 1
        return res
```

正确思路是动态规划。

和76题的区别在于，一个是子串，一个是子序列。

DP(i,j)表示T[:i]和S[:j]满足题意时S中的起始index，也就是子串W的位置为S[index:j]，index = DP(i,j)

所以就有递归方程：if T[i] == S[j]:DP(i,j) = DP(i-1,j-1) else:DP(i,j) = DP(i,j-1)

初始化：DP(i,j):if i == 0: DP(0,j) = j if j == 0: DP(i,0) = 0

由于题目要求的是子序列，所以需要得到起始位置和长度，还是最小窗口的。 也就是min(j-dp(len(T),j))，然后逐步更新即可。

```python
class Solution(object):
    def minWindow(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: str
        """
        m = len(T)
        n = len(S)
        dp = [[0] * (n+1) for _ in range(m+1)]
        for j in range(n+1):
            dp[0][j] = j + 1
            
        for i in range(1,m+1):
            for j in range(1,n+1):
                if T[i-1] == S[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = dp[i][j-1]
        
        start = 0
        l = n+1
        for j in range(1,n+1):
            if dp[m][j] :
                if j - dp[m][j] + 1 < l:
                    start = dp[m][j]-1
                    l = j - dp[m][j] + 1
        return "" if l == n+1 else S[start:start+l]
```

