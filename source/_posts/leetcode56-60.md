---

title: leetcode56-60
date: 2019-06-22 23:17:58
tags: leetcode
categories: 算法

---

## 1. 合并区间(Medium)

给出一个区间的集合，请合并所有重叠的区间。

**示例 1:**

```
输入: [[1,3],[2,6],[8,10],[15,18]]
输出: [[1,6],[8,10],[15,18]]
解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
```

**示例 2:**

```
输入: [[1,4],[4,5]]
输出: [[1,5]]
解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。
```

**解答：**

**思路：先排序，再合并。**

如果我们按照区间的 `start` 大小排序，那么在这个排序的列表中可以合并的区间一定是连续的。

1. 先按首位置进行排序;

2. 接下来,如何判断两个区间是否重叠呢?比如a = [1,4],b = [2,3]

3. 当a[1] >= b[0]说明两个区间有重叠.

4. 但是如何把这个区间找出来呢?

5. 左边位置一定是确定,就是a[0],而右边位置是max(a[1], b[1])

6. 所以,我们就能找出整个区间为:[1,4]

```python
class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        res = []
        for i in sorted(intervals,key = lambda x:x[0]):
            if res and i[0] <= res[-1][1]:
                res[-1][1] = max(res[-1][1],i[1])
            else:
                res.append(i)
        return res
```



## 2. 插入区间(Hard)

给出一个*无重叠的 ，*按照区间起始端点排序的区间列表。

在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

**示例 1:**

```
输入: intervals = [[1,3],[6,9]], newInterval = [2,5]
输出: [[1,5],[6,9]]
```

**示例 2:**

```
输入: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
输出: [[1,2],[3,10],[12,16]]
解释: 这是因为新的区间 [4,8] 与 [3,5],[6,7],[8,10] 重叠。
```

**解答：**

**思路：二分法**

1. 采用二分查找找到新插入取间[A,B]， A在所有左侧点中插入位置(或直接命中)，B在所有右侧点中插入位置（或直接命中）
2. 根据左右侧插入点构建结果

```python
class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        if not intervals:
            return [newInterval]
        
        def find(lis, k, i):
            lefti = 0
            leftj = len(lis) - 1

            while lefti <= leftj:
                mid = (lefti + leftj) // 2
                if lis[mid][i] > k:
                    leftj = mid - 1
                elif lis[mid][i] < k:
                    lefti = mid + 1
                else:
                    return mid
            return lefti

        left = find(intervals, newInterval[0], 0)
        right = find(intervals, newInterval[1], 1)
        
        l = max(left-1, 0)
        r = min(right+1, len(intervals))

        leftlis = intervals[:l]
        rightlis = intervals[r:]
        midlis = intervals[l:r]

        if midlis:
            if newInterval[0] <= midlis[0][1] and newInterval[1] >= midlis[-1][0]:
                midlis = [[min(midlis[0][0], newInterval[0]), max(midlis[-1][1], newInterval[1])]]
            elif newInterval[0] > midlis[0][1] and newInterval[1] >= midlis[-1][0]:
                midlis = [midlis[0], [newInterval[0], max(midlis[-1][1], newInterval[1])]]
            elif newInterval[0] <= midlis[0][1] and newInterval[1] < midlis[-1][0]:
                midlis = [[min(midlis[0][0], newInterval[0]), newInterval[1]], midlis[-1]]
            else:
                midlis = [midlis[0], newInterval, midlis[-1]]
        else:
            midlis = [newInterval]

        ans = leftlis + midlis + rightlis

        return ans
```

**思路二：**

也可以借鉴上一题的解法，不同在于，先把new插进去

```python
class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """
        if not intervals or len(intervals) == 0:
            return [newInterval]
        idx = -1
        for i in range(len(intervals)):
            if intervals[i].start > newInterval.start:
                idx = i
                break
        if idx != -1:
            intervals.insert(i, newInterval)
        else:
            intervals.append(newInterval)
        res = []
        for interval in intervals:
            if res and res[-1].end >= interval.start:
                res[-1].end = max(res[-1].end, interval.end)
            else:
                res.append(interval)
        return res
```



## 3. 最后一个单词的长度(Easy)

给定一个仅包含大小写字母和空格 `' '` 的字符串，返回其最后一个单词的长度。

如果不存在最后一个单词，请返回 0 。

**说明：**一个单词是指由字母组成，但不包含任何空格的字符串。

**示例:**

```
输入: "Hello World"
输出: 5
```

**解答：**

**思路：**

**很简单**

```python
class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        n = len(s)
        if n == 0:
            return 0
        res = 0
        flag = 0
        for i in range(n-1,-1,-1):
            if s[i] != ' ':
                res += 1
                flag = 1
            elif s[i] == ' ' and flag:
                return res
        return res
```

**思路二：**

直接找最后一个单词

先找最后一个单词最后一个字母

再找最后一个单词第一个字母

```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        end = len(s) - 1
        while end >= 0 and s[end] == " ":
            end -= 1
        if end == -1: return 0
        start = end
        while start >= 0 and s[start] != " ":
            start -= 1
        return end - start
```



## 4. 螺旋矩阵2(Medium)

给定一个正整数 *n*，生成一个包含 1 到 *n*2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵。

**示例:**

```
输入: 3
输出:
[
 [ 1, 2, 3 ],
 [ 8, 9, 4 ],
 [ 7, 6, 5 ]
]
```

**解答：**

**思路：**

这个题和螺旋矩阵题目很像。

**模拟顺时针转向**

假设数组有R 行 C 列，seen\[r\]\[c\]表示第 r 行第 c 列的单元格之前已经被访问过了。当前所在位置为 (r, c)，前进方向是 di。我们希望访问所有 R x C 个单元格。

当我们遍历整个矩阵，下一步候选移动位置是(next_r, next_c)。如果这个候选位置在矩阵范围内并且没有被访问过，那么它将会变成下一步移动的位置；否则，我们将前进方向顺时针旋转之后再计算下一步的移动位置。

```python
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        seen = [[False]*n for _ in range(n)]
        dr = [0,1,0,-1]
        dc = [1,0,-1,0]
        r,c,di = 0,0,0
        ans = [[-1]*n for _ in range(n)]
        for i in range(1,n*n+1):
            ans[r][c] = i
            seen[r][c] = True
            next_r,next_c = r+dr[di],c+dc[di]
            if 0<= next_r<n and 0<=next_c<n and not seen[next_r][next_c]:
                r = next_r
                c = next_c
            else:
                di = (di+1)%4
                r,c = r+dr[di],c+dc[di]
        return ans
```



## 5. 第K个排列(Medium)

给出集合 `[1,2,3,…,*n*]`，其所有元素共有 *n*! 种排列。

按大小顺序列出所有排列情况，并一一标记，当 *n* = 3 时, 所有排列如下：

1. `"123"`
2. `"132"`
3. `"213"`
4. `"231"`
5. `"312"`
6. `"321"`

给定 *n* 和 *k*，返回第 *k* 个排列。

**说明：**

- 给定 *n* 的范围是 [1, 9]。
- 给定 *k* 的范围是[1,  *n*!]。

**示例 1:**

```
输入: n = 3, k = 3
输出: "213"
```

**示例 2:**

```
输入: n = 4, k = 9
输出: "2314"
```

**解答：**

**思路一：回溯算法**

当然是暴力直接算出所有的排列然后取第k个，但是会超时

```python
class Solution(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        s = ''.join([str(i) for i in range(1, n+1)])
        print(s)
        def permunation(s):
            if len(s) == 0:
                return
            if len(s) == 1:
                return [s]
            res = []
            for i in range(len(s)):
                x = s[i]
                xs = s[:i] + s[i+1:]
                for j in permunation(xs):
                    res.append(x+j)
            return res
        return permunation(s)[k-1]
```

**思路二：找规律**

当有n位集合时，可以知道第一位显然有n个可能，而每一个第一位确定后，引申出来的可能性有 (n-1)! 个。

**所以反过来呢，第k个排列的第一位，就是 k/(n-1)! 余数记为mod。**

**于是第二位的答案也呼之欲出： mod/(n-2)!。**

这就是最核心的计算方法。下面讲解下实现的一些小细节。

1. 需要先定义从1到n的数组。上文中说的 k/(n-1)! 和 mod/(n-2)! 其实并不严谨，第二位实际上应该是算完第一位后排除第一位的答案后，剩余数组的第 mod/(n-2)! 个元素。
2. 由于引入了数组，第一位计算前mod 应该为 k-1。
3. 当余数为0时，实际上没有必要继续计算了，只需将剩余数组元素，依次添加进答案即可。

```python
class Solution(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        res = ''
        fac = [1] * (n+1)
        for i in range(1,n+1):
            fac[i] = fac[i-1]*i
        nums = [i for i in range(1,n+1)]
        k -= 1
        for i in range(1,n+1):
            idx = k/fac[n-i]
            res += str(nums[idx])
            nums.pop(idx)
            k -= idx * fac[n-i]
        return res
```

