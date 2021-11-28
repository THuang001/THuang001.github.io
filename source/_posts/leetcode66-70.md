---
title: leetcode66-70
date: 2019-06-25 20:35:17
tags: leetcode
categories: 算法
---

## 1. 加一(Easy)

给定一个由**整数**组成的**非空**数组所表示的非负整数，在该数的基础上加一。

最高位数字存放在数组的首位， 数组中每个元素只存储一个数字。

你可以假设除了整数 0 之外，这个整数不会以零开头。

**示例 1:**

```
输入: [1,2,3]
输出: [1,2,4]
解释: 输入数组表示数字 123。
```

**示例 2:**

```
输入: [4,3,2,1]
输出: [4,3,2,2]
解释: 输入数组表示数字 4321。
```

**解答：**

**思路一：递归**

这里是用的递归，很容易理解，如果空列表直接加1，最后一位小于9，那么直接就最后一位加1，否则添加一个0，然后再把余下的递归加1

```python
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        if not digits:
            return [1]
        if digits[-1] < 9:
            return digits[:-1]+[digits[-1]+1]
        else:
            return self.plusOne(digits[:-1])+[0]
```

**思路二：**

**迭代**

```python
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        carry = 1
        for i in range(len(digits)-1,-1,-1):
            digits[i] += carry
            if digits[i] < 10:
                carry = 0
                break
            else:
                digits[i] = 0
        return [1] + digits if carry == 1 else digits
```

**迭代用时比递归慢。。。。**

## 2. 二进制求和(Easy)

给定两个二进制字符串，返回他们的和（用二进制表示）。

输入为**非空**字符串且只包含数字 `1` 和 `0`。

**示例 1:**

```
输入: a = "11", b = "1"
输出: "100"
```

**示例 2:**

```
输入: a = "1010", b = "1011"
输出: "10101"
```

**解答：**

**思路：**

**这个就比较简单，递归。**

考虑是否进位。然后递归结果。

```python
class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        if a == '' or b == '':
            return a+b
        if a[-1] == '0' and b[-1] == '0':
            return self.addBinary(a[:-1],b[:-1]) +'0'
        elif a[-1] == '1' and b[-1] == '1':
            return self.addBinary(a[:-1],self.addBinary(b[:-1],'1')) + '0'
        else:
            return self.addBinary(a[:-1],b[:-1]) + '1'
```



## 3. 文本左右对齐(Hard)

给定一个单词数组和一个长度 *maxWidth*，重新排版单词，使其成为每行恰好有 *maxWidth* 个字符，且左右两端对齐的文本。

你应该使用“贪心算法”来放置给定的单词；也就是说，尽可能多地往每行中放置单词。必要时可用空格 `' '` 填充，使得每行恰好有 *maxWidth* 个字符。

要求尽可能均匀分配单词间的空格数量。如果某一行单词间的空格不能均匀分配，则左侧放置的空格数要多于右侧的空格数。

文本的最后一行应为左对齐，且单词之间不插入**额外的**空格。

**说明:**

- 单词是指由非空格字符组成的字符序列。
- 每个单词的长度大于 0，小于等于 *maxWidth*。
- 输入单词数组 `words` 至少包含一个单词。

**示例:**

```
输入:
words = ["This", "is", "an", "example", "of", "text", "justification."]
maxWidth = 16
输出:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]
```

**示例 2:**

```
输入:
words = ["What","must","be","acknowledgment","shall","be"]
maxWidth = 16
输出:
[
  "What   must   be",
  "acknowledgment  ",
  "shall be        "
]
解释: 注意最后一行的格式应为 "shall be    " 而不是 "shall     be",
     因为最后一行应为左对齐，而不是左右两端对齐。       
     第二行同样为左对齐，这是因为这行只包含一个单词。
```

**示例 3:**

```
输入:
words = ["Science","is","what","we","understand","well","enough","to","explain",
         "to","a","computer.","Art","is","everything","else","we","do"]
maxWidth = 20
输出:
[
  "Science  is  what we",
  "understand      well",
  "enough to explain to",
  "a  computer.  Art is",
  "everything  else  we",
  "do                  "
]
```

**解答：**

**思路：**

这道题确实难。

解答和思考过程参考题解部分。

**思路:**
		**首先要理顺题意,给定一堆单词,让你放在固定长度字符串里**

1. 两个单词之间至少有一个空格,如果单词加空格长度超过maxWidth,说明该单词放不下,比如示例1:当我们保证this is an 再加入example变成this is an example总长度超过maxWidth,所以这一行只能放下this is an 这三个单词;
2. this is an长度小于maxWidth,我们考虑分配空格,并保证左边空格数大于右边的
3. 最后一行,要尽量靠左,例如示例2的:"shall be "
   

我们针对上面三个问题,有如下解决方案.

1. 先找到一行最多可以容下几个单词;
2. 分配空格,例如this is an ,对于宽度为maxWidth,我们可以用maxWidth - all_word_len 与需要空格数商为 单词间 空格至少的个数,余数是一个一个分配给左边.就能保证左边空格数大于右边的.例如 16 - 8 = 8 ,a = 8 / 2, b = 8 % 2两个单词间要有4个空格,因为余数为零不用分配;
3. 针对最后一行单独考虑;



```python
class Solution(object):
    def fullJustify(self, words, maxWidth):
        """
        :type words: List[str]
        :type maxWidth: int
        :rtype: List[str]
        """
        res = []
        n = len(words)
        i = 0
        def one_row_words(i):
            left = i
            cur_row = len(words[i])
            i += 1
            while i < n:
                if cur_row + len(words[i]) + 1 > maxWidth:
                    break
                cur_row += len(words[i]) + 1
                i += 1
            return left,i
        while i < n:
            left,i = one_row_words(i)
            one_row = words[left:i]
            if i == n:
                res.append(' '.join(one_row).ljust(maxWidth,' '))
                break
            all_word_len = sum(len(i) for i in one_row)
            space_num = i - left - 1
            remain_space = maxWidth - all_word_len 
            if space_num:
                a,b = divmod(remain_space,space_num)
            tmp = ''
            for word in one_row:
                if tmp:
                    tmp += ' '*a
                    if b:
                        tmp += ' '
                        b -= 1
                tmp += word
            res.append(tmp.ljust(maxWidth,' '))
        return res
```



## 4. x的平方根(Easy)

实现 `int sqrt(int x)` 函数。

计算并返回 *x* 的平方根，其中 *x* 是非负整数。

由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

**示例 1:**

```
输入: 4
输出: 2
```

**示例 2:**

```
输入: 8
输出: 2
说明: 8 的平方根是 2.82842..., 
     由于返回类型是整数，小数部分将被舍去。
```

解答：

思路：很明显，就是二分法

```python
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if  x == 0:
            return 0
        if x == 1:
            return 1
        l ,r = 0,x-1
        while l <= r:
            mid = l +((r-l)>>1)
            if mid * mid <= x and (mid+1) * (mid +1) > x:
                return mid
            elif mid * mid <x:
                l = mid + 1
            else:
                r = mid - 1
```

**思路二：牛顿法**

![image.png](https://pic.leetcode-cn.com/36b76d291e8c934a5f1826f52f9f4f8b20c47e301e7c408123a43486a8c4e3dc-image.png)

```python
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        t = x
        while t * t >x:
            t = (t + x/t)/2
        return t
```



## 5. 爬楼梯(Easy)

假设你正在爬楼梯。需要 *n* 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

**注意：**给定 *n* 是一个正整数。

**示例 1：**

```
输入： 2
输出： 2
解释： 有两种方法可以爬到楼顶。
1.  1 阶 + 1 阶
2.  2 阶
```

**示例 2：**

```
输入： 3
输出： 3
解释： 有三种方法可以爬到楼顶。
1.  1 阶 + 1 阶 + 1 阶
2.  1 阶 + 2 阶
3.  2 阶 + 1 阶
```

解答：

思路：动态规划

很明显。dp(n)  =dp(n-1) + dp(n-2)

```python
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        a = 1
        b = 2
        if n == 1:
            return 1
        if n == 2:
            return 2
        for i in range(2,n):
            a,b = b,a+b
        return b
```

