---
title: leetcode71-75
date: 2019-06-25 20:38:51
tags: leetcode
categories: 算法
---

## 1. 简化路径(Medium)

以 Unix 风格给出一个文件的**绝对路径**，你需要简化它。或者换句话说，将其转换为规范路径。

在 Unix 风格的文件系统中，一个点（`.`）表示当前目录本身；此外，两个点 （`..`） 表示将目录切换到上一级（指向父目录）；两者都可以是复杂相对路径的组成部分。更多信息请参阅：[Linux / Unix中的绝对路径 vs 相对路径](https://blog.csdn.net/u011327334/article/details/50355600)

请注意，返回的规范路径必须始终以斜杠 `/` 开头，并且两个目录名之间必须只有一个斜杠 `/`。最后一个目录名（如果存在）**不能**以 `/` 结尾。此外，规范路径必须是表示绝对路径的**最短**字符串。

 

**示例 1：**

```
输入："/home/"
输出："/home"
解释：注意，最后一个目录名后面没有斜杠。
```

**示例 2：**

```
输入："/../"
输出："/"
解释：从根目录向上一级是不可行的，因为根是你可以到达的最高级。
```

**示例 3：**

```
输入："/home//foo/"
输出："/home/foo"
解释：在规范路径中，多个连续斜杠需要用一个斜杠替换。
```

**示例 4：**

```
输入："/a/./b/../../c/"
输出："/c"
```

**示例 5：**

```
输入："/a/../../b/../c//.//"
输出："/c"
```

**示例 6：**

```
输入："/a//b////c/d//././/.."
输出："/a/b/c"
```

**解答：**

**思路：**

非常简单的模拟题，利用一个栈来储存当前的路径。用 "/" 将输入的全路径分割成多个部分，对于每一个部分循环处理：如果为空或者 "." 则忽略，如果是 ".." ，则出栈顶部元素（如果栈为空则忽略），其他情况直接压入栈即可。

```python
class Solution(object):
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        stack = []
        for item in path.split('/'):
            if item and item != '.':
                if item == '..':
                    if stack:
                        stack.pop()
                else:
                    stack.append(item)
        if not stack:
            return '/'
        else:
            return '/' +'/'.join(stack)
```



## 2. 编辑距离(Hard)

给定两个单词 *word1* 和 *word2*，计算出将 *word1* 转换成 *word2* 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

1. 插入一个字符
2. 删除一个字符
3. 替换一个字符

**示例 1:**

```
输入: word1 = "horse", word2 = "ros"
输出: 3
解释: 
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```

**示例 2:**

```
输入: word1 = "intention", word2 = "execution"
输出: 5
解释: 
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
```

**解答：**

**思路：动态规划**

其实这个题很简单。如果你学过编辑距离的话。

**就是一个动态规划。**

**dp(i,j) =  min(dp(i-1,j)+1,dp(i,j-1)+1,dp(i-1,j-1 + tmp))** 

**Tmp = 0 if w[i] == s[j] else 1**

```python
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        n = len(word1)
        m = len(word2)
        if n == 0 or m == 0:
            return max(n,m)
        dp = [[i+j for j in range(m+1)] for i in range(n+1)]
        for i in range(1,n+1):
            for j in range(1,m+1):
                tmp = 0 if word1[i-1] == word2[j-1] else 1
                dp[i][j] = min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+tmp)
        return dp[-1][-1]
```



## 3. 矩阵置零(Medium)

给定一个 *m* x *n* 的矩阵，如果一个元素为 0，则将其所在行和列的所有元素都设为 0。请使用**原地**算法**。**

**示例 1:**

```
输入: 
[
  [1,1,1],
  [1,0,1],
  [1,1,1]
]
输出: 
[
  [1,0,1],
  [0,0,0],
  [1,0,1]
]
```

**示例 2:**

```
输入: 
[
  [0,1,2,0],
  [3,4,5,2],
  [1,3,1,5]
]
输出: 
[
  [0,0,0,0],
  [0,4,5,0],
  [0,3,1,0]
]
```

**进阶:**

- 一个直接的解决方案是使用  O(*m**n*) 的额外空间，但这并不是一个好的解决方案。
- 一个简单的改进方案是使用 O(*m* + *n*) 的额外空间，但这仍然不是最好的解决方案。
- 你能想出一个常数空间的解决方案吗？

**解答：**

**思路一：**

如果矩阵中任意一个格子有零我们就记录下它的行号和列号，这些行和列的所有格子在下一轮中全部赋为零。

**算法**

1. 我们扫描一遍原始矩阵，找到所有为零的元素。
2. 如果我们找到 [i, j] 的元素值为零，我们需要记录下行号 i 和列号 j。
3. 用两个 sets ，一个记录行信息一个记录列信息。
4. 最后，我们迭代原始矩阵，对于每个格子检查行 r 和列 c 是否被标记过，如果是就将矩阵格子的值设为 0。

```python
class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        row = len(matrix)
        col = len(matrix[0])
        rows,cols = set(),set()
        
        for i in range(row):
            for j in range(col):
                if matrix[i][j] == 0:
                    rows.add(i)
                    cols.add(j)
        for i in range(row):
            for j in range(col):
                if i in rows or j in cols:
                    matrix[i][j] = 0
```

思路二：常数空间

关键思想: **用matrix第一行和第一列记录该行该列是否有0,作为标志位**

但是对于第一行,和第一列要设置一个标志位,为了防止自己这一行(一列)也有0的情况.

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        row = len(matrix)
        col = len(matrix[0])
        row0_flag = False
        col0_flag = False
        # 找第一行是否有0
        for j in range(col):
            if matrix[0][j] == 0:
                row0_flag = True
                break
        # 第一列是否有0
        for i in range(row):
            if matrix[i][0] == 0:
                col0_flag = True
                break
        # 把第一行或者第一列作为 标志位
        for i in range(1, row):
            for j in range(1, col):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0
                    
        for i in range(1, row):
            for j in range(1, col):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        if row0_flag:
            for j in range(col):
                matrix[0][j] = 0
        if col0_flag:
            for i in range(row):
                matrix[i][0] = 0
```



## 4. 搜索二维矩阵(Medium)

编写一个高效的算法来判断 *m* x *n* 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

- 每行中的整数从左到右按升序排列。
- 每行的第一个整数大于前一行的最后一个整数。

**示例 1:**

```
输入:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 3
输出: true
```

**示例 2:**

```
输入:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 13
输出: false
```

**解答：**

**思路：很明显用二分法，不假思索。**

```python
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix or not matrix[0]:
            return False
        row = len(matrix)
        col = len(matrix[0])
        l,r = 0,row*col-1
        while l <= r:
            mid = (l+r)>>1
            value = matrix[mid//col][mid%col]
            if target == value:
                return True
            elif target < value:
                r = mid -1
            else:
                l = mid + 1
        return False
```



## 5. 颜色分类(Medium)

给定一个包含红色、白色和蓝色，一共 *n* 个元素的数组，**原地**对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

**注意:**
不能使用代码库中的排序函数来解决这道题。

**示例:**

```
输入: [2,0,2,1,1,0]
输出: [0,0,1,1,2,2]
```

**进阶：**

- 一个直观的解决方案是使用计数排序的两趟扫描算法。
  首先，迭代计算出0、1 和 2 元素的个数，然后按照0、1、2的排序，重写当前数组。
- 你能想出一个仅使用常数空间的一趟扫描算法吗？



**解答：**

**思路：**

**以前见过这道题，第一时间想到双指针，但是没办法记录当前指针位置。**

**所以很明显，三指针法。**

我们用三个指针（begin, end 和curr）来分别追踪0的最右边界，2的最左边界和当前考虑的元素。

这里是用三个指针，begin, cur, end，cur需要遍历整个数组

- cur 指向0，交换begin与cur， begin++,cur++
- cur 指向1，不做任何交换，cur++
- cur 指向2，交换end与cur，end--

```python
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        p0 = curr = 0
        p2 = len(nums) - 1
        while curr <= p2:
            if nums[curr] == 0:
                nums[p0],nums[curr] = nums[curr],nums[p0]
                p0 += 1
                curr += 1
            elif nums[curr] == 2:
                nums[p2],nums[curr] = nums[curr],nums[p2]
                p2 -= 1
            else:
                curr += 1
```

**几个易错点：**

1. **一定要按照这个顺序来比较，先比较0，再比较2，再比较1，具体可以灵活，需要自己领悟。**
2. **一定是if-elif-else的顺序**

**思路二：**

**考虑数字和index之间的关系，从而替换数字。**

**这个思路明显，很巧妙。**

```python
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        n0, n1, n2 = -1, -1, -1
        for i in range(len(nums)):
            if nums[i] == 0:
                n0, n1, n2 = n0+1, n1+1, n2+1
                nums[n2] = 2
                nums[n1] = 1
                nums[n0] = 0
            elif nums[i] == 1:
                n1, n2 = n1+1, n2+1
                nums[n2] = 2
                nums[n1] = 1
            else:
                n2 += 1
                nums[n2] = 2
```

