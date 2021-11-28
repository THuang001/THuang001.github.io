---
title: leetcode76-80
date: 2019-06-26 20:46:22
tags: leetcode
categories: 算法
---

## 1. 最小覆盖子串(Hard)

给你一个字符串 S、一个字符串 T，请在字符串 S 里面找出：包含 T 所有字母的最小子串。

**示例：**

```
输入: S = "ADOBECODEBANC", T = "ABC"
输出: "BANC"
```

**说明：**

- 如果 S 中不存这样的子串，则返回空字符串 `""`。
- 如果 S 中存在这样的子串，我们保证它是唯一的答案。

**解答：**

**思路：滑动窗口算法**

本问题要求我们返回字符串S 中包含字符串T的全部字符的最小窗口。我们称**包含T的全部字母的窗口为可行窗口。**

可以用**简单的滑动窗口法**来解决本问题。

**在滑动窗口类型的问题中都会有两个指针。一个用于延伸现有窗口的 right指针，和一个用于收缩窗口的left 指针。在任意时刻，只有一个指针运动，而另一个保持静止。**

本题的解法很符合直觉。我们通过移动right指针不断扩张窗口。当窗口包含全部所需的字符后，如果能收缩，我们就收缩窗口直到得到最小窗口。

答案是最小的可行窗口。

1. **初始，left指针和right指针都指向S的第一个元素.**

```python
        for c in t:
            lookup[c] += 1
        left = right = 0
```

2. **将 right 指针右移，扩张窗口，直到得到一个可行窗口，亦即包含T的全部字母的窗口。**

```python
				while right < len(s):
            if lookup[s[right]] > 0:
                count -= 1
            lookup[s[right]] -= 1
            right += 1
```



3. **得到可行的窗口后，将left指针逐个右移，若得到的窗口依然可行，则更新最小窗口大小。**

```python
						while count == 0:
                if min_len > right-left:
                    min_len = right-left
                    res = s[left:right]
                if lookup[s[left]] == 0:
                    count += 1
                lookup[s[left]] += 1
                left += 1
```



4. **若窗口不再可行，则跳转至 2。**

```python
        while right < len(s):
            if lookup[s[right]] > 0:
                count -= 1
            lookup[s[right]] -= 1
            right += 1
            while count == 0:
                if min_len > right-left:
                    min_len = right-left
                    res = s[left:right]
                if lookup[s[left]] == 0:
                    count += 1
                lookup[s[left]] += 1
                left += 1
```

结合起来，就是以下代码。

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
        left = right = 0
        min_len = float('inf')
        count = len(t)
        res = ''
        while right < len(s):
            if lookup[s[right]] > 0:
                count -= 1
            lookup[s[right]] -= 1
            right += 1
            while count == 0:
                if min_len > right-left:
                    min_len = right-left
                    res = s[left:right]
                if lookup[s[left]] == 0:
                    count += 1
                lookup[s[left]] += 1
                left += 1
        return res
```



## 2. 组合(Medium)

给定两个整数 *n* 和 *k*，返回 1 ... *n* 中所有可能的 *k* 个数的组合。

**示例:**

```
输入: n = 4, k = 2
输出:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```

**解答：**

**思路：回溯算法*

**回溯法** 是一种通过遍历所有可能成员来寻找全部可行解的算法。**若候选 不是 可行解 (或者至少不是 最后一个 解)，回溯法会在前一步进行一些修改以舍弃该候选，换而言之， 回溯 并再次尝试。**

这是一个回溯法函数。

**它将第一个添加到组合中的数和现有的组合作为参数。 backtrack(first, curr)**

1. 若组合完成- 添加到输出中。
2. 遍历从 first t到 n的所有整数。
   * 将整数 i 添加到现有组合 curr中。
   * 继续向组合中添加更多整数 : backtrack(i + 1, curr)。
   * 将 i 从 curr中移除，实现回溯。

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        def backtrack(first = 1, curr = []):
            if len(curr) == k:  
                output.append(curr[:])
            for i in range(first, n + 1):
                curr.append(i)
                backtrack(i + 1, curr)
                curr.pop()
        
        output = []
        backtrack()
        return output
```

**下面是我的解答，上面是官方的解答。**

**异曲同工。**

```python
class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        res = []
        def helper(i,k,tmp):
            if k == 0:
                res.append(tmp)
            for j in range(i,n+1):
                helper(j+1,k-1,tmp+[j])
        helper(1,k,[])
        return res
```



## 3. 子集(Medium)

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

**解答：**

**思路：回溯算法**

思考方式很简单，从0和[]出发，逐步加上每一个元素，递归式的扩展下去。

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

思路二：迭代

当然还有更加简单明显的迭代算法

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        for i in nums:
            res = res + [[i] + num for num in res]
        return res
```



## 4. 单词搜索(Medium)

给定一个二维网格和一个单词，找出该单词是否存在于网格中。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

**示例:**

```
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

给定 word = "ABCCED", 返回 true.
给定 word = "SEE", 返回 true.
给定 word = "ABCB", 返回 false.
```

**解答：**

**思路：回溯算法+DFS**

这道题一看就很简单，**回溯加DFS**，这两个几乎都是同时出现的。

```python
class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        if not board:
            return False
        if not word:
            return True
        row = len(board)
        col = len(board[0]) if row else 0
        def dfs(i,j,idx):
            if not 0<=i<row or not 0<=j<col or board[i][j] != word[idx]:
                return False
            if idx == len(word) - 1:
                return True
            board[i][j] = '*'
            res = dfs(i+1,j,idx+1) or dfs(i-1,j,idx+1) or dfs(i,j+1,idx+1) or dfs(i,j-1,idx+1)
            board[i][j] = word[idx]
            return res
        return any(dfs(i,j,0) for i in range(row) for j in range(col))
```



## 5. 删除排序数组中的重复项(Medium)

给定一个排序数组，你需要在**原地**删除重复出现的元素，使得每个元素最多出现两次，返回移除后数组的新长度。

不要使用额外的数组空间，你必须在**原地修改输入数组**并在使用 O(1) 额外空间的条件下完成。

**示例 1:**

```
给定 nums = [1,1,1,2,2,3],

函数应返回新长度 length = 5, 并且原数组的前五个元素被修改为 1, 1, 2, 2, 3 。

你不需要考虑数组中超出新长度后面的元素。
```

**示例 2:**

```
给定 nums = [0,0,1,1,1,1,2,3,3],

函数应返回新长度 length = 7, 并且原数组的前五个元素被修改为 0, 0, 1, 1, 2, 3, 3 。

你不需要考虑数组中超出新长度后面的元素。
```

**说明:**

为什么返回数值是整数，但输出的答案是数组呢?

请注意，输入数组是以**“引用”**方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。

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

**思路：**

```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i = 0
        for n in nums:
            if i < 2 or n != nums[i-2]:
                nums[i] = n
                i += 1
        return i
```

此题可以有通用模板，将2改成k的话：

```python
class Solution:
    def removeDuplicates(self, nums， k):
        i = 0
        for n in nums:
            if i < k or n != nums[i-k]:
                nums[i] = n
                i += 1
        return i
```

