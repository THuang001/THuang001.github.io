---
title: leetcode120-140
date: 2019-07-15 15:23:32
tags: leetcode
categories: 算法
---

## 1. 买卖股票的最佳时机(Easy)

给定一个数组，它的第 *i* 个元素是一支给定股票第 *i* 天的价格。

如果你最多只允许完成一笔交易（即买入和卖出一支股票），设计一个算法来计算你所能获取的最大利润。

注意你不能在买入股票前卖出股票。

**示例 1:**

```
输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
```

**示例 2:**

```
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```

解答：

思路：很简单的动态规划题目。

遍历数组找到当前最低价格，用当天价格减去最低价格获得最大利润

```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices or len(prices) == 0:
            return 0
        p,minv = 0,prices[0]
        for i in range(1,len(prices)):
            minv = min(minv,prices[i-1])
            p = max(p,prices[i]-minv)
        return p
```



## 2. 买卖股票的最佳时机2(Easy)

给定一个数组，它的第 *i* 个元素是一支给定股票第 *i* 天的价格。

设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

**注意：**你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

**示例 1:**

```
输入: [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
```

**示例 2:**

```
输入: [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
```

**示例 3:**

```
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```

 解答：

思路：也是动态规划，只不过是多了几次交易。

* 考虑买股票的策略：设今天价格p1，明天价格p2，若p1 < p2则今天买入明天卖出，赚取p2 - p1；
  * 若遇到连续上涨的交易日，第一天买最后一天卖收益最大，等价于每天买卖（因为没有交易手续费）；
  * 遇到价格下降的交易日，不买卖，因此永远不会亏钱。
* 赚到了所有交易日的钱，所有亏钱的交易日都未交易，理所当然会利益最大化。\

```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        p = 0
        for i in range(1,len(prices)):
            tmp = prices[i] - prices[i-1]
            if tmp > 0:
                p += tmp
        return p
```



## 3. 买卖股票的最佳时机3(Hard)

给定一个数组，它的第 *i* 个元素是一支给定的股票在第 *i* 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 *两笔* 交易。

**注意:** 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

**示例 1:**

```
输入: [3,3,5,0,0,3,1,4]
输出: 6
解释: 在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
     随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。
```

**示例 2:**

```
输入: [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。   
     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。   
     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
```

**示例 3:**

```
输入: [7,6,4,3,1] 
输出: 0 
解释: 在这个情况下, 没有交易完成, 所以最大利润为 0。
```

**解答：**

**思路：依旧是动态规划**

动态规划

dp\[k][i]到第i天经过k次交易得到最大的利润.

动态方程: dp\[k][i] = max(dp\[k][i-1], dp\[k-1][j-1] + prices[i] - prices[j]) 0 <=j <= i

```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """       
        if not prices: return 0
        n = len(prices)
        dp = [[0] * n for _ in range(3)]
        for k in range(1, 3):
            pre_max = -prices[0]
            for i in range(1, n):
                pre_max = max(pre_max, dp[k - 1][i - 1] - prices[i])
                dp[k][i] = max(dp[k][i - 1], prices[i] + pre_max)
        return dp[-1][-1]
```



## 4. 二叉树中最大路径和(Hard)

给定一个**非空**二叉树，返回其最大路径和。

本题中，路径被定义为一条从树中任意节点出发，达到任意节点的序列。该路径**至少包含一个**节点，且不一定经过根节点。

**示例 1:**

```
输入: [1,2,3]

       1
      / \
     2   3

输出: 6
```

**示例 2:**

```
输入: [-10,9,20,null,null,15,7]

   -10
   / \
  9  20
    /  \
   15   7

输出: 42
```

**解答：**

**思路：**

根据题意，最大路径和可能出现在：

- 左子树中
- 右子树中
- 包含根节点与左右子树

我们的思路是递归从bottom向top`return`的过程中，记录`左子树和右子树中路径更大的那个`，并向父节点提供`当前节点和子树组成的最大值`。

递归设计：

- 返回值：

  ```python
  (root.val) + max(left, right)
  ```

  即此节点与左右子树最大值之和，较差的解直接被舍弃，不会再被用到。

  - 需要注意的是，若计算结果`tmp <= 0`，意味着对根节点有`负贡献`，不会在任何情况选这条路（父节点中止），因此返回`0`。

- 递归终止条件：越过叶子节点，返回`0`；

- 记录最大值：当前节点`最大值 = root.val + left + right`。

最终返回所有路径中的全局最大值即可。

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.res = float('-inf')
        def helper(root):
            if not root:
                return 0
            left = helper(root.left)
            right = helper(root.right)
            self.res = max(self.res,left+right+root.val)
            return max(max(left,right)+root.val,0)
        helper(root)
        return self.res
```



## 5. 验证回文串(Easy)

给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。

**说明：**本题中，我们将空字符串定义为有效的回文串。

**示例 1:**

```
输入: "A man, a plan, a canal: Panama"
输出: true
```

**示例 2:**

```
输入: "race a car"
输出: false
```

**解答：**

**思路：回文串问题，双指针法**

```python
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if not s:
            return True
        left = 0
        right = len(s)-1
        while left < right:
            while left < right and not s[left].isalnum():
                left += 1
            while left < right and not s[right].isalnum():
                right -= 1
            if s[left].lower() != s[right].lower():
                return False
            left += 1
            right -= 1
        return True
```



## 6. 单词接龙2(Hard)

给定两个单词（*beginWord* 和 *endWord*）和一个字典 *wordList*，找出所有从 *beginWord* 到 *endWord* 的最短转换序列。转换需遵循如下规则：

1. 每次转换只能改变一个字母。
2. 转换过程中的中间单词必须是字典中的单词。

**说明:**

- 如果不存在这样的转换序列，返回一个空列表。
- 所有单词具有相同的长度。
- 所有单词只由小写字母组成。
- 字典中不存在重复的单词。
- 你可以假设 *beginWord* 和 *endWord* 是非空的，且二者不相同。

**示例 1:**

```
输入:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

输出:
[
  ["hit","hot","dot","dog","cog"],
  ["hit","hot","lot","log","cog"]
]
```

**示例 2:**

```
输入:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

输出: []

解释: endWord "cog" 不在字典中，所以不存在符合要求的转换序列。
```

**解答：**

**思路：DFS + BFS**

用BFS求从beginWord 到endWord最短距离,经过哪些单词, 用字典记录离beginWord的距离;

用DFS求从beginWord 到endWord有哪些路径.

```python
class Solution(object):
    def findLadders(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: List[List[str]]
        """
        wordList = set(wordList)
        res = []
        from collections import defaultdict
        next_word_dict = defaultdict(list)
        distance = {}
        distance[beginWord] = 0
        
        def next_word(word):
            ans = []
            for i in range(len(word)):
                for j in range(97,123):
                    tmp = word[:i] + chr(j) + word[i+1:]
                    if tmp != word and tmp in wordList:
                        ans.append(tmp)
            return ans
        def bfs():
            step = 0
            flag = False
            cur = [beginWord]
            while cur:
                step += 1
                next_time = []
                for word in cur:
                    for nw in next_word(word):
                        next_word_dict[word].append(nw)
                        if nw == endWord:
                            flag = True
                        if nw not in distance:
                            distance[nw] = step
                            next_time.append(nw)
                if flag:
                    break
                cur = next_time
        def dfs(tmp,step):
            if tmp[-1] == endWord:
                res.append(tmp)
                return
            for word in next_word_dict[tmp[-1]]:
                if distance[word] == step + 1:
                    dfs(tmp + [word],step + 1)
        
        bfs()
        dfs([beginWord],0)
        return res
```



当然，这道题一看，标准的回溯法，基本的BFS，典型的level order traverse

有两个坑：

1. 不要判断字典里的某两个word是否只相差一个字母，而是要判断某个word的邻居（和他只相差一个字母的所有word）是否在字典里，这样的改进会使这一步的复杂度下降很多，否则超时妥妥
2. 每一轮访问过的word一定要从字典中删除掉，否则一定会超时

最后见到end word就收
完成

拿题目的例子来看：

```\
        hit

         |

        hot

       /   \

      dot   lot

       |     |

      dog   log

        \   /

         cog
```

routine 字典，然后再根据这个来寻找路径

'cog': ['log', 'dog']`这里的意思就是说在走到`'cog'`之前尝试过了`'log'`和`'dog'```，即previous tried node

而生成字典的过程就是BFS的，此处保证寻找的路径就是最短的。

```python
class Solution(object):
    def findLadders(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: List[List[str]]
        """
        def backtrack(res,routine,path,endWord):
            if len(routine[endWord]) == 0:
                res.append([endWord] + path)
            else:
                for pre in routine[endWord]:
                    backtrack(res,routine,[endWord] + path,pre)
            
            
        lookup = set(wordList) | set([beginWord])
        res,cur,routine = [],set([beginWord]),{word:[] for word in lookup}
        while cur and endWord not in cur:
            next_queue = set()
            for word in cur:
                lookup.remove(word)
            for word in cur:
                for i in range(len(word)):
                    for j in range(97,123):
                        tmp = word[:i] + chr(j) + word[i+1:]
                        if tmp in lookup:
                            next_queue.add(tmp)
                            routine[tmp].append(word)
            cur = next_queue
            
        if cur:
            backtrack(res,routine,[],endWord)
        return res
```



## 7. 单词接龙(Medium)

给定两个单词（*beginWord* 和 *endWord*）和一个字典，找到从 *beginWord*到 *endWord* 的最短转换序列的长度。转换需遵循如下规则：

1. 每次转换只能改变一个字母。
2. 转换过程中的中间单词必须是字典中的单词。

**说明:**

- 如果不存在这样的转换序列，返回 0。
- 所有单词具有相同的长度。
- 所有单词只由小写字母组成。
- 字典中不存在重复的单词。
- 你可以假设 *beginWord* 和 *endWord* 是非空的，且二者不相同。

**示例 1:**

```
输入:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

输出: 5

解释: 一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog",
     返回它的长度 5。
```

**示例 2:**

```
输入:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

输出: 0

解释: endWord "cog" 不在字典中，所以无法进行转换。
```

解答：

思路：

类似于层次遍历，画成树结构就很好看了，求树的深度。

```python
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        res = 1
        cur = [beginWord]
        if endWord not in wordList:
            return 0
        wordList = set(wordList)
        while cur:
            next_time = []
            if endWord in cur:
                return res
            for word in cur:
                if word in wordList:
                    wordList.remove(word)
                for i in range(len(word)):
                    for j in range(97,123):
                        tmp = word[:i] + chr(j) + word[i+1:]
                        if tmp != word and tmp in wordList:
                            next_time.append(tmp)
            cur = next_time
            res += 1
        return 0
```



## 8. 最长连续序列(Hard)

给定一个未排序的整数数组，找出最长连续序列的长度。

要求算法的时间复杂度为 *O(n)*。

**示例:**

```
输入: [100, 4, 200, 1, 3, 2]
输出: 4
解释: 最长连续序列是 [1, 2, 3, 4]。它的长度为 4。
```

解答：

思路：

这道题思路很巧妙，判断每个数后续是否在数组中

```python
class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums = set(nums)
        res = 0
        for x in nums:
            if x-1 not in nums:
                y = x + 1
                while y in nums:
                    y += 1
                res = max(res,y-x)
        return res
```



## 9. 求根到叶子结点数字之和(Medium)

给定一个二叉树，它的每个结点都存放一个 `0-9` 的数字，每条从根到叶子节点的路径都代表一个数字。

例如，从根到叶子节点路径 `1->2->3` 代表数字 `123`。

计算从根到叶子节点生成的所有数字之和。

**说明:** 叶子节点是指没有子节点的节点。

**示例 1:**

```
输入: [1,2,3]
    1
   / \
  2   3
输出: 25
解释:
从根到叶子节点路径 1->2 代表数字 12.
从根到叶子节点路径 1->3 代表数字 13.
因此，数字总和 = 12 + 13 = 25.
```

**示例 2:**

```
输入: [4,9,0,5,1]
    4
   / \
  9   0
 / \
5   1
输出: 1026
解释:
从根到叶子节点路径 4->9->5 代表数字 495.
从根到叶子节点路径 4->9->1 代表数字 491.
从根到叶子节点路径 4->0 代表数字 40.
因此，数字总和 = 495 + 491 + 40 = 1026.
```

解答：

思路：DFS遍历



```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.res = 0
        def helper(root,tmp):
            if not root:
                return
            if not root.left and not root.right:
                self.res += int(tmp + str(root.val))
            helper(root.left,tmp+str(root.val))
            helper(root.right,tmp+str(root.val))
        helper(root,"")
        return self.res
```



## 10. 被围绕的区域(Medium)

给定一个二维的矩阵，包含 `'X'` 和 `'O'`（**字母 O**）。

找到所有被 `'X'` 围绕的区域，并将这些区域里所有的 `'O'` 用 `'X'` 填充。

**示例:**

```
X X X X
X O O X
X X O X
X O X X
```

运行你的函数后，矩阵变为：

```
X X X X
X X X X
X X X X
X O X X
```

**解释:**

被围绕的区间不会存在于边界上，换句话说，任何边界上的 `'O'` 都不会被填充为 `'X'`。 任何不在边界上，或不与边界上的 `'O'` 相连的 `'O'` 最终都会被填充为 `'X'`。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。

**解答：**

**思路：**

反向找答案，先把非围绕区域的"O"找出来，然后把这些区域标记，最后将剩余的"O"换成X，标记区域换回去，表示为"O"

```python
class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        if not board or not board[0]:
            return 
        row = len(board)
        col = len(board[0])
        
        def dfs(i,j):
            board[i][j] == 'B'
            for x,y in [(-1,0),(1,0),(0,1),(0,-1)]:
                tmp_i = i + x
                tmp_j = j + y
                if 1 <= tmp_i < row and 1 <= tmp_j < col and board[tmp_i][tmp_j] =='O':
                    dfs(tmp_i,tmp_j)
        for j in range(col):
            if board[0][j] == 'O':
                dfs(0,j)
            if board[row-1][j] == 'O':
                dfs(row-1,j)
        for i in range(row):
            if board[i][0] == 'O':
                dfs(i,0)
            if board[i][col-1] == 'O':
                dfs(i,col-1)
        for i in range(row):
            for j in range(col):
                if board[i][j] == 'O':
                    board[i][j] == 'X'
                if board[i][j] == 'B':
                    board[i][j] == 'O'
```



## 11. 分割回文串(Medium)

给定一个字符串 *s*，将 *s* 分割成一些子串，使每个子串都是回文串。

返回 *s* 所有可能的分割方案。

**示例:**

```
输入: "aab"
输出:
[
  ["aa","b"],
  ["a","a","b"]
]
```

解答：

思路：

很标准的回溯法，针对可能的每一个分割点，如果前面是回文，则递归判断后面。

```python
class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        res = []        
        def helper(s,tmp):
            if not s:
                res.append(tmp)
            for i in range(1,len(s)+1):
                if s[:i] == s[:i][::-1]:
                    helper(s[i:],tmp + [s[:i]])
        helper(s,[])
        return res
```



## 12. 分割回文串2(Hard)

给定一个字符串 *s*，将 *s* 分割成一些子串，使每个子串都是回文串。

返回符合要求的最少分割次数。

**示例:**

```
输入: "aab"
输出: 1
解释: 进行一次分割就可将 s 分割成 ["aa","b"] 这样两个回文子串。
```



解答：

思路：

如果还是用上一题的思路，结果超时。

```python
class Solution(object):
    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        ans = float('inf')
        if s == s[::-1]:
            return 0
        for i in range(1,len(s)+1):
            if s[:i] == s[:i][::-1]:
                ans = min(self.minCut(s[i:])+1,ans)
        return ans
```

所以，使用动态规划。

很容易理解。

如果s[j:i]是回文，那么cut[i] =min(cut[i],cut[j-1]+1)

怎么判断回文，用DP数组。

```python
class Solution(object):
    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        cut = list(range(len(s)))
        dp = [[False] * len(s) for _ in range(len(s))]
        for i in range(len(s)):
            for j in range(i+1):
                if s[i] == s[j] and(i-j<2 or dp[j+1][i-1]):
                    dp[j][i] = True
                    if j == 0:
                        cut[i] = 0
                    else:
                        cut[i] = min(cut[i],cut[j-1]+1)
        return cut[-1]
```



## 13. 克隆图(Medium)

给定无向[**连通**](https://baike.baidu.com/item/连通图/6460995?fr=aladdin)图中一个节点的引用，返回该图的[**深拷贝**](https://baike.baidu.com/item/深拷贝/22785317?fr=aladdin)（克隆）。图中的每个节点都包含它的值 `val`（`Int`） 和其邻居的列表（`list[Node]`）。

**示例：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/02/23/113_sample.png)

```
输入：
{"$id":"1","neighbors":[{"$id":"2","neighbors":[{"$ref":"1"},{"$id":"3","neighbors":[{"$ref":"2"},{"$id":"4","neighbors":[{"$ref":"3"},{"$ref":"1"}],"val":4}],"val":3}],"val":2},{"$ref":"4"}],"val":1}

解释：
节点 1 的值是 1，它有两个邻居：节点 2 和 4 。
节点 2 的值是 2，它有两个邻居：节点 1 和 3 。
节点 3 的值是 3，它有两个邻居：节点 2 和 4 。
节点 4 的值是 4，它有两个邻居：节点 1 和 3 。
```

 

**提示：**

1. 节点数介于 1 到 100 之间。
2. 无向图是一个[简单图](https://baike.baidu.com/item/简单图/1680528?fr=aladdin)，这意味着图中没有重复的边，也没有自环。
3. 由于图是无向的，如果节点 *p* 是节点 *q* 的邻居，那么节点 *q* 也必须是节点 *p* 的邻居。
4. 必须将**给定节点的拷贝**作为对克隆图的引用返回。

**解答：**

**思路：**：典型的DFS或者BFS

DFS方法

```python
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, neighbors):
        self.val = val
        self.neighbors = neighbors
"""
class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        lookup = {}
        def dfs(node):
            if not node:
                return
            if node in lookup:
                return lookup[node]
            clone = Node(node.val,[])
            lookup[node] = clone
            for n in node.neighbors:
                clone.neighbors.append(dfs(n))
            return clone
        return dfs(node)
```

BFS方法

```python
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, neighbors):
        self.val = val
        self.neighbors = neighbors
"""
class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        from collections import deque
        lookup = {}
        def bfs(node):
            if not node:
                return
            clone = Node(node.val,[])
            lookup[node] = clone
            queue = deque()
            queue.appendleft(node)
            while queue:
                tmp = queue.pop()
                for n in tmp.neighbors:
                    if n not in lookup:
                        lookup[n] = Node(n.val,[])
                        queue.appendleft(n)
                    lookup[tmp].neighbors.append(lookup[n])
            return clone
        return bfs(node)
```



## 14. 加油站(Medium)

在一条环路上有 *N* 个加油站，其中第 *i* 个加油站有汽油 `gas[i]` 升。

你有一辆油箱容量无限的的汽车，从第 *i* 个加油站开往第 *i+1* 个加油站需要消耗汽油 `cost[i]` 升。你从其中的一个加油站出发，开始时油箱为空。

如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。

**说明:** 

- 如果题目有解，该答案即为唯一答案。
- 输入数组均为非空数组，且长度相同。
- 输入数组中的元素均为非负数。

**示例 1:**

```
输入: 
gas  = [1,2,3,4,5]
cost = [3,4,5,1,2]

输出: 3

解释:
从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
因此，3 可为起始索引。
```

**示例 2:**

```
输入: 
gas  = [2,3,4]
cost = [3,4,3]

输出: -1

解释:
你不能从 0 号或 1 号加油站出发，因为没有足够的汽油可以让你行驶到下一个加油站。
我们从 2 号加油站出发，可以获得 4 升汽油。 此时油箱有 = 0 + 4 = 4 升汽油
开往 0 号加油站，此时油箱有 4 - 3 + 2 = 3 升汽油
开往 1 号加油站，此时油箱有 3 - 3 + 3 = 3 升汽油
你无法返回 2 号加油站，因为返程需要消耗 4 升汽油，但是你的油箱只有 3 升汽油。
因此，无论怎样，你都不可能绕环路行驶一周。
```

解答：

思路：贪心算法

每一次到达一个加油站，更新total和cur，如果cur<0，说明到达不了下一站，以下一站为起点重新开始。

贪心在于以下一点为起始。

**从上一次重置的加油站到当前加油站的任意一个加油站出发，到达当前加油站之前， `cur` 也一定会比 0 小**

```python
class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        total = cur = 0
        start = 0
        for i in range(len(gas)):
            total += gas[i] - cost[i]
            cur += gas[i] - cost[i]
            if cur < 0:
                start = i+1
                cur = 0
        return start if total>=0 else -1
```



## 15. 分发糖果(Hard)

老师想给孩子们分发糖果，有 *N* 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。

你需要按照以下要求，帮助老师给这些孩子分发糖果：

- 每个孩子至少分配到 1 个糖果。
- 相邻的孩子中，评分高的孩子必须获得更多的糖果。

那么这样下来，老师至少需要准备多少颗糖果呢？

**示例 1:**

```
输入: [1,0,2]
输出: 5
解释: 你可以分别给这三个孩子分发 2、1、2 颗糖果。
```

**示例 2:**

```
输入: [1,2,2]
输出: 4
解释: 你可以分别给这三个孩子分发 1、2、1 颗糖果。
     第三个孩子只得到 1 颗糖果，这已满足上述两个条件。
```

解答：

思路：

- 首先我们从左往右边看，如果当前child的rating比它左边child的rating大，那么当前child分的candy肯定要比它左边多1
- 然后从右往左边看，如果当前child的rating比它右边child的rating大，那么当前child分的candy肯定要比它右边多1

```python
class Solution(object):
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        n = len(ratings)
        res = [1] * n
        for i in range(1,n):
            if ratings[i] > ratings[i-1]:
                res[i] = res[i-1]+1
        for i in range(n-2,-1,-1):
            if ratings[i] > ratings[i+1]:
                res[i] = max(res[i],res[i+1]+1)
        return sum(res)
```

还有一种空间复杂度更低的做法

就是寻找连续下降的长度，然后加上连续下降长度所需的糖果数，同时更新之前的糖果值

最终的是下面的pre和des

以及怎么计算res

```python
class Solution(object):
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        if not ratings:
            return 0
        res = 1
        pre = 1
        des = 0
        for i in range(1,len(ratings)):
            if ratings[i] >= ratings[i-1]:
                if des > 0:
                    res += (1 + des) * des/2
                    if pre <= des:
                        res += des-pre+1
                    des=0
                    pre=1
                if ratings[i] == ratings[i-1]:
                    pre = 1
                else:
                    pre += 1
                res += pre
            else:
                des += 1
        if des > 0:
            res += (1 + des) * des/2
            if pre <= des:
                res += des-pre + 1
        return res
```

## 16. 只出现一次的数字(Medium)

给定一个**非空**整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

**说明：**

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

**示例 1:**

```
输入: [2,2,1]
输出: 1
```

**示例 2:**

```
输入: [4,1,2,1,2]
输出: 4
```

解答：

思路：

就很简单，用异或，剑指offer原题

```python
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums or len(nums) == 0:
            return
        tmp = nums[0]
        for i in range(1,len(nums)):
            tmp ^= nums[i]
        return tmp
```



## 17. 只出现一次的数字2(Medium)

给定一个**非空**整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。

**说明：**

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

**示例 1:**

```
输入: [2,2,3,2]
输出: 3
```

**示例 2:**

```
输入: [0,1,0,1,0,1,99]
输出: 99
```

解答：

思路：

首先我们会定义两个变量ones和twos，当遍历nums的时候，对于重复元素x，第一次碰到x的时候，我们会将x赋给ones，第二次碰到后再赋给twos，第三次碰到就全部消除。赋值和消除的动作可以通过xor很简单的实现。所以我们就可以写出这样的代码

```
 ones = (ones^num)
 twos = (twos^num)
```

但是上面写法忽略了，只有当ones是x的时候，我们会将0赋给twos，那要怎么做呢？我们知道如果b=0，那么b^num就变成了x，而x&~x就完成了消除操作.所以代码应该写成：

```
 ones = (ones^num)&~(twos)
 twos = (twos^num)&~(ones)
```

第一次出现x记录在ones中，并且此时twos应为0；第二次出现x记录在twos中，同时ones置为0,；第三次出现x，则ones，twos均重置为0

例如：第一个数：10，则ones = 10，twos = 0,第二个数10，则ones = 0, twos = 10, 第三个数10, 则ones= 0,twos=0

```python
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        one,two = 0,0
        for i in nums:
            one = (one^i)&~two
            two = (two^i)&~one
        return one
```

若题目改成找只出现两次的数，则return twos



**思路二：按位求和**

长度为32的数组，将每个数按照二进制位1或者0求和，最后与1异或还原

## 18. 复制带随机指针的链表(Hard)

给定一个链表，每个节点包含一个额外增加的随机指针，该指针可以指向链表中的任何节点或空节点。

要求返回这个链表的**深拷贝**。 

 

**示例：**

**![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/02/23/1470150906153-2yxeznm.png)**

```
输入：
{"$id":"1","next":{"$id":"2","next":null,"random":{"$ref":"2"},"val":2},"random":{"$ref":"2"},"val":1}

解释：
节点 1 的值是 1，它的下一个指针和随机指针都指向节点 2 。
节点 2 的值是 2，它的下一个指针指向 null，随机指针指向它自己。
```

 

**提示：**

1. 你必须返回**给定头的拷贝**作为对克隆列表的引用。

解答：

思路：

这个题是剑指offer原题，有好几种做法

一种是先复制，再拆分，在拼接的过程

一种是用哈希表映射原节点和新节点。

```python
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, next, random):
        self.val = val
        self.next = next
        self.random = random
"""
class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        if not head:
            return
        d,node = {None:None},head
        while node:
            d[node] = Node(node.val,None,None)
            node = node.next
        node = head
        while node:
            d[node].next = d[node.next]
            d[node].random = d[node.random]
            node = node.next
        return d[head]
```



## 19. 单词拆分(Medium)

给定一个**非空**字符串 *s* 和一个包含**非空**单词列表的字典 *wordDict*，判定 *s* 是否可以被空格拆分为一个或多个在字典中出现的单词。

**说明：**

- 拆分时可以重复使用字典中的单词。
- 你可以假设字典中没有重复的单词。

**示例 1：**

```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
```

**示例 2：**

```
输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     注意你可以重复使用字典中的单词。
```

**示例 3：**

```
输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false
```

思路：

比较愚蠢的动态规划思想，当然回溯法肯定会超时

```python
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """        
  			dp = [False] * (len(s)+1)
        dp[0] = True
        for i in range(len(s)):
            if dp[i]:
                for j in range(i+1,len(s)+1):
                    if s[i:j] in wordDict:
                        dp[j] = True
        return dp[-1]
```

还有另一种更加高效的回溯算法，在wordDict最长长度内寻找

```python
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """        
        if not s or not wordDict:return False
        n=len(s)    
        dp=[False]*(n)
#         优化 先找到字典中单词的最大长度
        max_len=max(map(len,wordDict))            
        for i in range(n):
#             在最大长度内遍历即可
            start=max(-1,i-max_len)
            for j in range(start,i+1):
                if j==-1 or dp[j]==True:
                    if s[j+1:i+1] in wordDict:
                        dp[i]=True
                        break
        return dp[-1]
```



## 20. 单词拆分2(Hard)

给定一个**非空**字符串 *s* 和一个包含**非空**单词列表的字典 *wordDict*，在字符串中增加空格来构建一个句子，使得句子中所有的单词都在词典中。返回所有这些可能的句子。

**说明：**

- 分隔时可以重复使用字典中的单词。
- 你可以假设字典中没有重复的单词。

**示例 1：**

```
输入:
s = "catsanddog"
wordDict = ["cat", "cats", "and", "sand", "dog"]
输出:
[
  "cats and dog",
  "cat sand dog"
]
```

**示例 2：**

```
输入:
s = "pineapplepenapple"
wordDict = ["apple", "pen", "applepen", "pine", "pineapple"]
输出:
[
  "pine apple pen apple",
  "pineapple pen apple",
  "pine applepen apple"
]
解释: 注意你可以重复使用字典中的单词。
```

**示例 3：**

```
输入:
s = "catsandog"
wordDict = ["cats", "dog", "sand", "and", "cat"]
输出:
[]
```

解答：

思路：

这道题就是一个很简单的DFS，也就是深度回溯算法

但是很可惜，超时。

```python
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        res = []
        def dfs(idx,tmp):
            if idx == len(s):
                res.append(tmp[:-1])
            for i in range(idx+1,len(s)+1):
                if s[idx:i] in wordDict:
                    dfs(i,tmp+s[idx:i]+' ')
        dfs(0,'')
        return res
```

另一种思路是，用cache加速后的回溯算法，不会超时。

```python
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        def helper(s,wordDict,memo):
            if s in memo:
                return memo[s]
            if not s:
                return []
            res = []
            for word in wordDict:
                if not s.startswith(word):
                    continue
                if len(s) == len(word):
                    res.append(word)
                else:
                    re = helper(s[len(word):],wordDict,memo)
                    for item in re:
                        item = word + ' ' + item
                        res.append(item)
            memo[s] = res
            return res
        return helper(s,wordDict,{})
```

