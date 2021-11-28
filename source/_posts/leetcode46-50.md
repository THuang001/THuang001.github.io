---
title: leetcode46-50
date: 2019-06-22 15:33:27
tags: leetcode
categories: 算法
---

## 1. 全排列(Medium)

给定一个**没有重复**数字的序列，返回其所有可能的全排列。

**示例:**

```
输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```

**解答：**

**思路一：**

这个题说简单也简单，说难也难。

主要思想还是回溯，就看你理解回溯算法程度深不深了。

代码里很好的解释了回溯算法。

```python
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        def backtrack(nums,tmp):
            if not nums:
                res.append(tmp)
                return
            for i in range(len(nums)):
                backtrack(nums[:i]+nums[i+1:],tmp+[nums[i]])
        backtrack(nums,[])
        return res
```

**执行用时 :28 ms, 在所有 Python 提交中击败了96.79%的用户**

当然，如果你觉得这个不好理解，还有更容易理解的方法。

**思路二：**

每次取一个作为prefix, 剩下的继续做permutation，然后连接起来加入res中

```python
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) == 0:
            return []
        if len(nums) == 1:
            return [nums]
        res = []
        for i in range(len(nums)):
            prefix = nums[i]
            rest = nums[:i] + nums[i+1:]
            for j in self.permute(rest):
                res.append([prefix]+j)
        return res
```



## 2. 全排列2(Medium)


给定一个可包含重复数字的序列，返回所有不重复的全排列。

**示例:**

```
输入: [1,1,2]
输出:
[
  [1,1,2],
  [1,2,1],
  [2,1,1]
]
```

**解答：**

**思路：**

这一题怎么说呢，和上一题一模一样，唯一就是不能重复，所以，稍微修改一下上一题的代码就可以AC了

```python
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) == 0:
            return []
        if len(nums) == 1:
            return [nums]
        
        res = []
        for i in range(len(nums)):
            prefix = nums[i]
            rest = nums[:i]+nums[i+1:]
            for j in self.permuteUnique(rest):
                if [prefix] +j not in res:
                    res.append([prefix] + j)
        return res
```



## 3. 旋转图像(Medium)

给定一个 *n* × *n* 的二维矩阵表示一个图像。

将图像顺时针旋转 90 度。

**说明：**

你必须在**原地**旋转图像，这意味着你需要直接修改输入的二维矩阵。**请不要**使用另一个矩阵来旋转图像。

**示例 1:**

```
给定 matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

原地旋转输入矩阵，使其变为:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]
```

**示例 2:**

```
给定 matrix =
[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
], 

原地旋转输入矩阵，使其变为:
[
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]
```

**解答：**

**思路：**

这个题想明白翻转过程就OK了

我们可以先沿对角线翻转，然后对每一行翻转

```python
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        row = len(matrix)
        col = len(matrix[0])
        for i in range(row):
            for j in range(i,col):
                matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]
        for i in range(row):
            matrix[i].reverse()
```



## 4. 字母异位词分组(Medium)

给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

**示例:**

```
输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
```

**说明：**

- 所有输入均为小写字母。
- 不考虑答案输出的顺序。

**解答：**

**思路：**

这个题可以先将每个str字典序排序，比较是否一样，然后按key存入字典里。

```python
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        maps = {}
        for i in strs:
            tmp = ''.join(sorted(list(i)))
            if tmp in maps:
                maps[tmp].append(i)
            else:
                maps[tmp] = [i]
        return maps.values()
```



## 5. Pow(x,n)(Medium)

实现 pow(*x*, *n*) ，即计算 x 的 n 次幂函数。

**示例 1:**

```
输入: 2.00000, 10
输出: 1024.00000
```

**示例 2:**

```
输入: 2.10000, 3
输出: 9.26100
```

**示例 3:**

```
输入: 2.00000, -2
输出: 0.25000
解释: 2-2 = 1/22 = 1/4 = 0.25
```

**说明:**

- -100.0 < *x* < 100.0
- *n* 是 32 位有符号整数，其数值范围是 [−231, 231 − 1] 。

**解答：**

**思路：**

这个题就很简单了，剑指offer原题

```python
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n < 0:
            x = 1/x
            n = -n
        res = 1
        while n:
            if n & 1:
                res *= x
            x *= x
            n >>= 1
        return res
```

也可以用递归，递归的坏处在于可能发生递归调用栈溢出

```python
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0:
            return 1
        if n < 0:
            return self.myPow(1/x,-n)
        if n & 1:
            return x * self.myPow(x*x,n>>1)
        else:
            return self.myPow(x*x,n>>1)
```

可能与测试用例有关。。。递归比迭代要快。