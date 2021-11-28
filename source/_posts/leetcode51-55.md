---
title: leetcode51-55
date: 2019-06-22 23:17:50
tags: leetcode
categories: 算法
---

## 1. N皇后(Hard)

*n* 皇后问题研究的是如何将 *n* 个皇后放置在 *n*×*n* 的棋盘上，并且使皇后彼此之间不能相互攻击。

![](/images/8-queens.png)

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

**解答：**

**思路：回溯算法**

这个题很难，大一就觉得好难。。。

在建立算法之前，我们来考虑两个有用的细节。

1. 一行只可能有一个皇后且一列也只可能有一个皇后。

2. 这意味着没有必要再棋盘上考虑所有的方格。只需要按列循环即可。

3. 对于所有的主对角线有 行号 + 列号 = 常数，对于所有的次对角线有 行号 - 列号 = 常数.

4. 这可以让我们标记已经在攻击范围下的对角线并且检查一个方格 (行号, 列号) 是否处在攻击位置。

现在已经可以写回溯函数 backtrack(row = 0).

* 从第一个 row = 0 开始.

* 循环列并且试图在每个 column 中放置皇后.

  * 如果方格 (row, column) 不在攻击范围内
    * 在 (row, column) 方格上放置皇后。
    * 排除对应行，列和两个对角线的位置。
    *  If 所有的行被考虑过，row == N
      *   意味着我们找到了一个解
    * ​    Else
      * 继续考虑接下来的皇后放置 backtrack(row + 1).
    * ​    回溯：将在 (row, column) 方格的皇后移除.

  

```python
class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        def could_place(row,col):
            return not (cols[col] + hill[row - col] + dale[row + col])
        def place_queen(row,col):
            queens.add((row,col))
            cols[col] = 1
            hill[row - col] = 1
            dale[row + col] = 1
        def remove_queen(row,col):
            queens.remove((row,col))
            cols[col] = 0
            hill[row - col] = 0
            dale[row + col] = 0
        def add_solution():
            solution = []
            for _,col in sorted(queens):
                solution.append('.'*col+'Q'+'.'*(n-col-1))
            output.append(solution)
        def backtrack(row = 0):
            for col in range(n):
                if could_place(row,col):
                    place_queen(row,col)
                    if row + 1 == n:
                        add_solution()
                    else:
                        backtrack(row + 1)
                    remove_queen(row,col)
        cols = [0] * n
        hill = [0] * (2*n-1)
        dale = [0] * (2*n-1)
        queens = set()
        output = []
        backtrack()
        return output
```

**思路二：DFS**

八皇后问题可以推广为更一般的n皇后摆放问题：这时棋盘的大小变为n×n，而皇后个数也变成n。当且仅当n = 1或n ≥ 4时问题有解。

对于任意(x,y),如果要让新的点和它不能处于同一条横行、纵行或斜线上，则新点(p,q)必须要满足p+q != x+y 和p-q!= x-y, 前者针对左下右上斜线，后者针对左上右下斜线，两者同时都保证了不在同一条横行和纵行上。

代码中变量的含义:

- cols_lst: 每一行皇后的column位置组成的列表
- cur_row：目前正在判断的row的index
- xy_diff：所有x-y组成的列表
- xy_sum：所有x+y组成的列表

```python
class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        def dfs(cols_lst, xy_diff, xy_sum):
            cur_row = len(cols_lst)
            if cur_row == n:
                ress.append(cols_lst)
            for col in range(n):
                if col not in cols_lst and cur_row - col not in xy_diff and cur_row + col not in xy_sum:
                    dfs(cols_lst+[col], xy_diff+[cur_row-col], xy_sum+[cur_row+col])
        ress = []
        dfs([], [], [])
        return [['.' * i + 'Q' + '.' * (n-i-1) for i in res] for res in ress]
```



## 2. N皇后2(Hard)

*n* 皇后问题研究的是如何将 *n* 个皇后放置在 *n*×*n* 的棋盘上，并且使皇后彼此之间不能相互攻击。

![](/images/8-queens.png)

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

**解答：**

**思路：**

和上题完全一模一样，只是最后输出的时候，输出的是结果的长度。

```python
class Solution(object):
    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: int
        """
        def dfs(col_list,xy_diff,xy_sum):
            cur_row = len(col_list)
            if cur_row == n:
                ress.append(col_list)
            for col in range(n):
                if col not in col_list and cur_row-col not in xy_diff and cur_row+col not in xy_sum:
                    dfs(col_list+[col],xy_diff+[cur_row-col],xy_sum+[cur_row+col])
        ress = []
        dfs([],[],[])
        return len(ress)
```



## 3. 最大子序和(Easy)


给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**示例:**

```
输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

**进阶:**

如果你已经实现复杂度为 O(*n*) 的解法，尝试使用更为精妙的分治法求解。

**解答：**

**思路：**

这道题很简单，动态规划，逐步更新局部最优解和全局最优解。

```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        cur_sum,all_sum = nums[0],nums[0]
        for i in range(1,len(nums)):
            cur_sum = max(cur_sum+nums[i],nums[i])
            all_sum = max(all_sum,cur_sum)
        return all_sum
```



## 4. 螺旋矩阵(Medium)

给定一个包含 *m* x *n* 个元素的矩阵（*m* 行, *n* 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素。

**示例 1:**

```
输入:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
输出: [1,2,3,6,9,8,7,4,5]
```

**示例 2:**

```
输入:
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
输出: [1,2,3,4,8,12,11,10,9,5,6,7]
```

**解答：**

**思路：**

绘制螺旋轨迹路径，我们发现当路径超出界限或者进入之前访问过的单元格时，会顺时针旋转方向。

**算法**

假设数组有R 行 C 列，seen\[r\]\[c\]表示第 r 行第 c 列的单元格之前已经被访问过了。当前所在位置为 (r, c)，前进方向是 di。我们希望访问所有 R x C 个单元格。

当我们遍历整个矩阵，下一步候选移动位置是(next_r, next_c)。如果这个候选位置在矩阵范围内并且没有被访问过，那么它将会变成下一步移动的位置；否则，我们将前进方向顺时针旋转之后再计算下一步的移动位置。

```python
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if not matrix:
            return []
        R = len(matrix)
        C = len(matrix[0])
        seen = [[False]*C for _ in matrix]
        dr = [0,1,0,-1]
        dc = [1,0,-1,0]
        r = c = di = 0
        ans = []
        for _ in range(R*C):
            ans.append(matrix[r][c])
            seen[r][c] = True
            next_r,next_c = r+dr[di],c+dc[di]
            if 0<= next_r < R and 0<= next_c < C and not seen[next_r][next_c]:
                r = next_r
                c = next_c
            else:
                di = (di+1)%4
                r = r + dr[di]
                c = c + dc[di]
        return ans
```



## 5. 跳跃游戏(Medium)

给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个位置。

**示例 1:**

```
输入: [2,3,1,1,4]
输出: true
解释: 从位置 0 到 1 跳 1 步, 然后跳 3 步到达最后一个位置。
```

**示例 2:**

```
输入: [3,2,1,0,4]
输出: false
解释: 无论怎样，你总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ， 所以你永远不可能到达最后一个位置。
```

**解答：**

**思路：**

这个题很好想。

就是从后往前倒推，如果倒数第二个能到底倒数第一个位置，那么可以就去求是否可以达到倒数第二个位置。

就是反复往回推的过程。

```python
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        last = len(nums)-1
        for i in range(len(nums)-1,-1,-1):
            if i + nums[i] >= last:
                last = i
        return last == 0
```

