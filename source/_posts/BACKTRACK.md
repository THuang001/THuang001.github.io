---
title: 回溯算法类题目
date: 2019-07-10 21:33:00
tags: backtrack
categories: 算法
---

# 一、综述

回溯是一种通过穷举所有可能情况来找到所有解的算法。如果一个候选解最后被发现并不是可行解，回溯算法会舍弃它，并在前面的一些步骤做出一些修改，并重新尝试找到可行解。

# 二、回溯算法思路

所谓Backtracking都是这样的思路：

**在当前局面下，你有若干种选择。**

**那么尝试每一种选择。**

**如果已经发现某种选择肯定不行（因为违反了某些限定条件），就返回；**

**如果某种选择试到最后发现是正确解，就将其加入解集**

所以你思考递归题时，只要明确三点就行：**选择 (Options)，限制 (Restraints)，结束条件 (Termination)。**即“ORT原则”。



比如对于括号生成这道题来说：

对于这道题，在任何时刻，你都有**两种选择**：

1. 加左括号。
2. 加右括号。

同时有以下**限制**：

1. 如果左括号已经用完了，则不能再加左括号了。
2. 如果已经出现的右括号和左括号一样多，则不能再加右括号了。因为那样的话新加入的右括号一定无法匹配。

**结束条件**是：
左右括号都已经用完。

**结束后的正确性**：
左右括号用完以后，一定是正确解。因为1. 左右括号一样多，2. 每个右括号都一定有与之配对的左括号。因此一旦结束就可以加入解集（有时也可能出现结束以后不一定是正确解的情况，这时要多一步判断）。

**递归函数传入参数**：
限制和结束条件中有“用完”和“一样多”字样，因此你需要知道左右括号的数目。
当然你还需要知道当前局面sublist和解集res。



# 三、例题

## 1. 电话号码的组合

给定一个仅包含数字 `2-9` 的字符串，返回所有它能表示的字母组合。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

![img](/images/17_telephone_keypad.png)

**示例:**

```
输入："23"
输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
```

**说明:**
尽管上面的答案是按字典序排列的，但是你可以任意选择答案输出的顺序。

### 思路：

这个就是标准的回溯算法了。

helper()函数，满足就res.append()，不满足就继续往下遍历。

```python
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        lookup = {'2':['a','b','c'],
                   '3':['d','e','f'],
                   '4':['g','h','i'],
                   '5':['j','k','l'],
                   '6':['m','n','o'],
                   '7':['p','q','r','s'],
                   '8':['t','u','v'],
                   '9':['w','x','y','z']}
        
        if not digits:
            return []
        n = len(digits)
        res = []
        def helper(i,tmp):
            if i == n:
                res.append(tmp)
                return
            for al in lookup[digits[i]]:
                helper(i+1,tmp+al)
        helper(0,"")
        return res
```

## 2. 括号生成

给出 *n* 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且**有效的**括号组合。

例如，给出 *n* = 3，生成结果为：

```
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
```

### 思路：

很明显的回溯算法。

为什么要判断left_p<right_p：因为在任意位置，左括号的数量是大于等于右括号的数量的。如果left<right，说明左括号小于右括号，则不满足。

```python
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        res = []
        
        def helper(left_p,right_p,tmp):
            if left_p == n and right_p == n:
                res.append(tmp)
                return
            if left_p > n or right_p > n or left_p < right_p:
                return
            helper(left_p+1,right_p,tmp+'(')
            helper(left_p,right_p+1,tmp+')')
        helper(0,0,"")
        return res
```

## 3. 解数独

编写一个程序，通过已填充的空格来解决数独问题。

一个数独的解法需**遵循如下规则**：

1. 数字 `1-9` 在每一行只能出现一次。
2. 数字 `1-9` 在每一列只能出现一次。
3. 数字 `1-9` 在每一个以粗实线分隔的 `3x3` 宫内只能出现一次。

空白格用 `'.'` 表示。

![img](http://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Sudoku-by-L2G-20050714.svg/250px-Sudoku-by-L2G-20050714.svg.png)

一个数独。

![img](http://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Sudoku-by-L2G-20050714_solution.svg/250px-Sudoku-by-L2G-20050714_solution.svg.png)

答案被标成红色。

**Note:**

- 给定的数独序列只包含数字 `1-9` 和字符 `'.'` 。
- 你可以假设给定的数独只有唯一解。
- 给定数独永远是 `9x9` 形式的。

### 思路：

经典的回溯算法。

对于每一个为'.'的点都从1走到9，如果valid就继续走，如果不valid就立马返回。

```python
class Solution(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        self.backtrack(board)
        
    def backtrack(self,board):
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for c in '123456789':
                        if self.isPointValid(board,i,j,c):
                            board[i][j] = c
                            if self.backtrack(board):
                                return True
                            else:
                                board[i][j] = '.'
                    return False
        return True
        
    def isPointValid(self,board,x,y,c):
        for i in range(9):
            if board[i][y] == c:
                return False
            if board[x][i] == c:
                return False
            if board[(x//3)*3+i//3][(y//3)*3+i%3] == c:
                return False
        return True
```

## 4. 组合总和

给定一个**无重复元素**的数组 `candidates` 和一个目标数 `target` ，找出 `candidates` 中所有可以使数字和为 `target` 的组合。

`candidates` 中的数字可以无限制重复被选取。

**说明：**

- 所有数字（包括 `target`）都是正整数。
- 解集不能包含重复的组合。 

**示例 1:**

```
输入: candidates = [2,3,6,7], target = 7,
所求解集为:
[
  [7],
  [2,2,3]
]
```

**示例 2:**

```
输入: candidates = [2,3,5], target = 8,
所求解集为:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
```

### 思路：

通过这道题，我们可以得到回溯算法的模板。

```python
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        candidates.sort()
        n = len(candidates)
        res = []
        
        def helper(i,tmp_sum,tmp_list):
            if i == n or tmp_sum > target:
                return
            if tmp_sum == target:
                res.append(tmp_list)
                return
            for j in range(i,n):
                if tmp_sum + candidates[j] > target:
                  	 break
                helper(j,tmp_sum + candidates[j],tmp_list + [candidates[j]])
        helper(0,0,[])
        return res                 
```

## 5. 组合总和2

给定一个数组 `candidates` 和一个目标数 `target` ，找出 `candidates` 中所有可以使数字和为 `target` 的组合。

`candidates` 中的每个数字在每个组合中只能使用一次。

**说明：**

- 所有数字（包括目标数）都是正整数。
- 解集不能包含重复的组合。 

**示例 1:**

```
输入: candidates = [10,1,2,7,6,1,5], target = 8,
所求解集为:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
```

**示例 2:**

```
输入: candidates = [2,5,2,1,2], target = 5,
所求解集为:
[
  [1,2,2],
  [5]
]
```

### 思路：

这道题和上一题一样，也是回溯，只不过区别在于，回溯的时候，跳过相同数字。

```python
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        candidates.sort()
        n = len(candidates)
        res = []
        def helper(i,tmp_sum,tmp):
            if tmp_sum == target:
                res.append(tmp)
                return
            for j in range(i,n):
                if tmp_sum + candidates[j] > target:
                    break
                if j > i and candidates[j] == candidates[j-1]:
                    continue
                helper(j+1,tmp_sum+candidates[j],tmp+[candidates[j]])
        helper(0,0,[])
        return res
```

## 6. 全排列

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

### 思路：

这样的题就很习惯了，终止条件也可以为not nums

```python
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        n = len(nums)
        def helper(nums,tmp):
            if len(tmp) == n:
                res.append(tmp)
            for i in range(len(nums)):
                helper(nums[:i]+nums[i+1:],tmp+[nums[i]])
        helper(nums,[])
        return res
```

## 7. 全排列2

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

### 思路：

和上一题一样的思路，只不过要跳过重复的组合

```python
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        res = []
        n = len(nums)
        visited = set()
        def helper(nums,tmp):
            if len(tmp) == n:
                res.append(tmp)
                return
            for i in range(len(nums)):
                if i in visited or (i>0 and i-1 not in visited and nums[i] == nums[i-1]):
                    continue
                visited.add(i)
                helper(nums,tmp+[nums[i]])
                visited.remove(i)
        helper(nums,[])
        return res
```

当然也有别的回溯的写法

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

## 8. N皇后

*n* 皇后问题研究的是如何将 *n* 个皇后放置在 *n*×*n* 的棋盘上，并且使皇后彼此之间不能相互攻击。

![img](/images/8-queens.png)

上图为 8 皇后问题的一种解法。

给定一个整数 *n*，返回所有不同的 *n* 皇后问题的解决方案。

每一种解法包含一个明确的 *n* 皇后问题的棋子放置方案，该方案中 `'Q'` 和 `'.'` 分别代表了皇后和空位。

**示例:**

```
输入: 4
输出: [
 [".Q..",  // 解法 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // 解法 2
  "Q...",
  "...Q",
  ".Q.."]
]
解释: 4 皇后问题存在两个不同的解法。
```

### 思路：

标准的回溯思想

```python
class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        res = []
        s = '.' * n
        def helper(i,tmp,col,z_dia,f_dia):
            if i  == n:
                res.append(tmp)
                return
            for j in range(n):
                if j not in col and (i+j) not in z_dia and (i-j) not in f_dia:
                    helper(i+1,tmp+[s[:j]+'Q'+s[j+1:]],col | {j},z_dia | {i+j},f_dia | {i-j})
        helper(0,[],set(),set(),set())
        return res
```



## 9. N皇后2

*n* 皇后问题研究的是如何将 *n* 个皇后放置在 *n*×*n* 的棋盘上，并且使皇后彼此之间不能相互攻击。

![img](/images/8-queens.png)

上图为 8 皇后问题的一种解法。

给定一个整数 *n*，返回 *n* 皇后不同的解决方案的数量。

**示例:**

```
输入: 4
输出: 2
解释: 4 皇后问题存在如下两个不同的解法。
[
 [".Q..",  // 解法 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // 解法 2
  "Q...",
  "...Q",
  ".Q.."]
]
```



### 思路：

```python
class Solution(object):
    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: int
        """
        self.res = 0
        def helper(i,col,z_dia,f_dia):
            if i  == n:
                return True
            for j in range(n):
                if j not in col and (i+j) not in z_dia and (i-j) not in f_dia:
                    if helper(i+1,col | {j},z_dia | {i+j},f_dia | {i-j}):
                        self.res += 1
            return False
        helper(0,set(),set(),set())
        return self.res
```

## 10. 子集

给定一组**不含重复元素**的整数数组 *nums*，返回该数组所有可能的子集（幂集）。

**说明：**解集不能包含重复的子集。

**示例:**

```
输入: nums = [1,2,3]
输出:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]
```

### 思路：

很标准的回溯模板了

```python
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        n = len(nums)
        def helper(i,tmp):
            res.append(tmp)
            for j in range(i,n):
                helper(j+1,tmp+[nums[j]])
        helper(0,[])
        return res
```



## 11. 子集2

给定一个可能包含重复元素的整数数组 ***nums***，返回该数组所有可能的子集（幂集）。

**说明：**解集不能包含重复的子集。

**示例:**

```
输入: [1,2,2]
输出:
[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]
```

### 思路：

很标准的回溯算法模板

```python
class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        nums.sort()
        def helper(i,tmp):
            res.append(tmp)
            for j in range(i,len(nums)):
                if j > i and nums[j] == nums[j-1]:
                    continue
                helper(j+1,tmp+[nums[j]])
        helper(0,[])
        return res
```

