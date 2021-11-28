---
title: leetcode4-6
date: 2019-06-18 23:31:45
tags: leetcode
categories: 算法
---

## 1. 寻找两个有序数组的中位数(Hard)

给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。

请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。

你可以假设 nums1 和 nums2 不会同时为空。

**示例 1:**

```
nums1 = [1, 3]
nums2 = [2]

则中位数是 2.0
```

**示例 2:**

```
nums1 = [1, 2]
nums2 = [3, 4]

则中位数是 (2 + 3)/2 = 2.5
```

**解答：**

**思路一：时间复杂度$O((m+n)*log(m+n))$，空间复杂度**$O(m+n)$

首先最简单粗暴的方法，就是我们将两个数字列表合并起来，排好序，找到中间的`median`就ok了，但是千万要注意一点，如果`median`有两个，需要算平均。

```python
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        nums = sorted(nums1 + nums2)
        if len(nums) % 2 == 1:
            return nums[len(nums)//2]
        else:
            return (nums[len(nums)//2-1] + nums[len(nums)//2]) / 2.0
```

**思路二：时间复杂度$O(log(m+n))$，空间复杂度**$O(1)$

这时候我们观察到题目给的一个条件，`nums1`和`nums2`本身也是有序的，放着这个条件不用反而用思路一是不是有点浪费了？换句话说我们没必要把他们整个排序，于是我们可以把它转化成经典的 [findKth问题](https://wizardforcel.gitbooks.io/the-art-of-programming-by-july/content/02.01.html)。

首先转成求`A`和`B`数组中第`k`小的数的问题, 然后用`k//2`在`A`和`B`中分别找。

比如 `k = 6`, 分别看`A`和`B`中的第`3`个数, 已知 `A1 <= A2 <= A3 <= A4 <= A5...`和 `B1 <= B2 <= B3 <= B4 <= B5...`, 如果`A3 <＝ B3`, 那么第`6`小的数肯定不会是`A1, A2, A3`, 因为最多有两个数小于`A1`(B1, B2), 三个数小于`A2`(A1, B1, B2), 四个数小于`A3`(A1, A2, B1, B2)。 关键点是从`k//2` 开始来找。那就可以排除掉A1, A2, A3, 转成求`A4, A5, ... B1, B2, B3, ...`这些数中第`3`小的数的问题, `k`就被减半了。

当`k == 1`或某一个数组空了, 这两种情况都是终止条件。

```python
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        def findKth(A,B,k):
            if len(A) == 0:
                return B[k-1]
            if len(B) == 0:
                return A[k-1]
            if k == 1:
                return min(A[0],B[0])
            
            a = A[k//2-1] if len(A) >= k // 2 else None
            b = B[k//2-1] if len(B) >= k // 2 else None
            
            # if a is None: 
            #     return findKth(A, B[k // 2:], k - k // 2) # 这里要注意：因为 k//2 不一定等于 (k - k//2)
            # if b is None:
            #     return findKth(A[k // 2:], B, k - k // 2)
            # if a < b:
            #     return findKth(A[k // 2:], B, k - k // 2)
            # else:
            #     return findKth(A, B[k // 2:], k - k // 2)

            if b is None or (a is not None and a < b):
                return findKth(A[k // 2:], B, k - k // 2)
            return findKth(A, B[k // 2:], k - k // 2) 
        
        n = len(nums1) + len(nums2)
        if n % 2 == 1:
            return findKth(nums1,nums2,n//2+1)
        else:
            small = findKth(nums1,nums2,n//2)
            large = findKth(nums1,nums2,n//2+1)
            return (small + large) / 2.0
```



**思路三：时间复杂度$O(log(m+n))$，空间复杂度**$O(1)$

`findKth` 函数我们可以用双指针的方式实现

```python
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        def findKth(A, pa, B, pb, k):
            res = 0
            m = 0
            while pa < len(A) and pb < len(B) and m < k:
                if A[pa] < B[pb]:
                    res = A[pa]
                    pa += 1
                else:
                    res = B[pb]
                    pb += 1
                m += 1
            while pa < len(A) and m < k:
                res = A[pa]
                pa += 1
                m += 1
            while pb < len(B) and m < k:
                res = B[pb]
                pb += 1
                m += 1
            return res
        n = len(nums1) + len(nums2)
        if n % 2 == 1:
            return findKth(nums1, 0, nums2, 0, n // 2 + 1)
        else:
            smaller = findKth(nums1, 0, nums2, 0, n // 2)
            bigger = findKth(nums1, 0, nums2, 0, n // 2 + 1)
            return (smaller + bigger) / 2.0
```

## 2. 最长回文字串(Medium)

给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。

**示例 1：**

```
输入: "babad"
输出: "bab"
注意: "aba" 也是一个有效答案。
```


**示例 2：**

```
输入: "cbbd"
输出: "bb"
```

**解答：**

**思路一：时间复杂度$O(n^2)$，空间复杂度$O(n^2)​**$

动态规划思想：dp\[i\]\[j\]表示s\[i:j+1\]是否是palindrome

```python
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        res = ''
        dp = [[False] * len(s) for i in range(len(s))]
        for i in range(len(s),-1,-1):
            for j in range(i,len(s)):
                dp[i][j] = (s[i] == s[j]) and (j-i<3 or dp[i+1][j-1])
                if dp[i][j] and j-i+1>len(res):
                    res = s[i:j+1]
        return res
```

**思路二：时间复杂度$O(n^2)$，空间复杂度$O(1)​**$

回文字符串长度为奇数和偶数是不一样的：

1. 奇数：`'xxx s[i] xxx'`, 比如 `'abcdcba'`
2. 偶数：`'xxx s[i] s[i+1] xxx'`, 比如 `'abcddcba'`

我们区分回文字符串长度为奇数和偶数的情况，然后依次把每一个字符当做回文字符串的中间字符，向左右扩展到满足回文的最大长度，不停更新满足回文条件的最长子串的左右`index`: `l` 和`r`，最后返回`s[l:r+1]`即为结果。

```python
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        l = 0 # left index of the current substring
        r = 0 # right index of the current substring
        max_len = 0 # length of the longest palindromic substring for now
        n = len(s)
        for i in range(n):
            # odd case: 'xxx s[i] xxx', such as 'abcdcba' 
            for j in range(min(i+1, n-i)): # 向左最多移动 i 位，向右最多移动 (n-1-i) 位
                if s[i-j] != s[i+j]: # 不对称了就不用继续往下判断了
                    break
                if 2 * j + 1 > max_len: # 如果当前子串长度大于目前最长长度
                    max_len = 2 * j + 1
                    l = i - j
                    r = i + j

            # even case: 'xxx s[i] s[i+1] xxx', such as 'abcddcba' 
            if i+1 < n and s[i] == s[i+1]:
                for j in range(min(i+1, n-i-1)): # s[i]向左最多移动 i 位，s[i+1]向右最多移动 [n-1-(i+1)] 位
                    if s[i-j] != s[i+1+j]: # 不对称了就不用继续往下判断了
                        break
                    if 2 * j + 2 > max_len:
                        max_len = 2 * j + 2
                        l = i - j
                        r = i + 1 + j
        return s[l:r+1]
```

**思路三：时间复杂度$O(n)$，空间复杂度$O(n)​**$

[Manacher算法](https://www.felix021.com/blog/read.php?2040)

[Useful link](https://leetcode-cn.com/problems/longest-palindromic-substring/solution/zhong-xin-kuo-san-dong-tai-gui-hua-by-liweiwei1419/)



## 3. Z字形变换(Medium)

将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。

比如输入字符串为 "LEETCODEISHIRING" 行数为 3 时，排列如下：

```python
L   C   I   R
E T O E S I I G
E   D   H   N
```

之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："LCIRETOESIIGEDHN"。

请你实现这个将字符串进行指定行数变换的函数：

```
string convert(string s, int numRows);
```

**示例 1:**

```
输入: s = "LEETCODEISHIRING", numRows = 3
输出: "LCIRETOESIIGEDHN"
```


**示例 2:**

```
输入: s = "LEETCODEISHIRING", numRows = 4
输出: "LDREOEIIECIHNTSG"
解释:

L     D     R
E   O E   I I
E C   I H   N
T     S     G

```

**解答：**

**思路一：时间复杂度$O(n)$，空间复杂度$O(n)​**$

`idx`从`0`开始，自增直到`numRows-1`，此后又一直自减到`0`，重复执行。

给个例子容易懂一些：`s = “abcdefghijklmn”`, `numRows = 4`

```
a    g    n 
b  f h  l   m 
c e  i k
d    j
```

从第一行开始往下，走到第四行又往上走，这里用 `step = 1` 代表往下走， `step = -1`代表往上走

因为只会有一次遍历，同时把每一行的元素都存下来，所以时间复杂度和空间复杂度都是 `O(N)`

```python
class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows == 1 or len(s) <= numRows:
            return s
        
        res = [''] * numRows
        idx,step = 0,1
        
        for c in s:
            res[idx] += c
            if idx == 0:
                step = 1
            elif idx == numRows - 1:
                step = -1
            idx += step
        return ''.join(res)
```

**思路二：模拟过程**

Z字形，就是两种状态，一种垂直向下，还有一种斜向上

控制好边界情况就可以了

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if not s:
            return ""
        if numRows == 1:return s
        s_Rows = [""] * numRows
        i  = 0
        n = len(s)
        while i < n:
            for j in range(numRows):
                if i < n:
                    s_Rows[j] += s[i]
                    i += 1
            for j in range(numRows-2,0,-1):
                if i < n:
                    s_Rows[j] += s[i]
                    i += 1
        return "".join(s_Rows)
```

