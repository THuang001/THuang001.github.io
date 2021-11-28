---
title: leetcode41-45(附415)
date: 2019-06-22 15:33:20
tags: leetcode
categories: 算法
---

## 1. 缺失的第一个正数(Hard)

给定一个未排序的整数数组，找出其中没有出现的最小的正整数。

**示例 1:**

```
输入: [1,2,0]
输出: 3
```

**示例 2:**

```
输入: [3,4,-1,1]
输出: 2
```

**示例 3:**

```
输入: [7,8,9,11,12]
输出: 1
```

**说明:**

你的算法的时间复杂度应为O(*n*)，并且只能使用常数级别的空间。

**解答：**

**思路：**

官方题解：还是哈希表，不过，key是nums的索引，value是正负号，如果nums[i]出现过，则为正号，否则为负号，nums[i]因为映射到索引，所以是按顺序来的。

```python
class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if 1 not in nums:
            return 1
        if len(nums) == 1:
            return 2
        for i in range(len(nums)):
            if nums[i]<= 0 or nums[i] > len(nums):
                nums[i] = 1
            
        for i in range(len(nums)):
            a = abs(nums[i])
            if a == len(nums):
                nums[0] = -abs(nums[0])
            else:
                nums[a] = -abs(nums[a])
        
        for i in range(1,len(nums)):
            if nums[i]>0:
                return i
        if nums[0] > 0:
            return len(nums)
        return len(nums)+1   
```



## 2. 接雨水(Hard)

给定 *n* 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

![](/images/rainwatertrap.png)

上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。

**示例:**

```
输入: [0,1,0,2,1,0,1,3,2,1,2,1]
输出: 6
```

**解答：**

**思路：**

题目有几个特性可用，bar width = 1,然后第一个和最后一个是不能trap water，其次中间的部分能trap多少水是看`左右高度差较低的那个 - 本身的高度`

设置双指针l和r，然后获取左右最低的那个高度min_h，因为短板效应，所以水的高度不可能大于min_h，所以我们继续移动双指针，如果接下来移动的位置的高度小于min_h，则可以装水，直到遇到高度高于min_h，更新l和r继续这个过程。

```python
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        l,r = 0,len(height)-1
        water = 0
        min_height = 0
        while l < r:
            min_height = min(height[l],height[r])
            while l < r and height[l] <= min_height:
                water += min_height - height[l]
                l += 1
            while l < r and height[r] <= min_height:
                water += min_height - height[r]
                r -= 1
        return water 
```



## 3. 字符串相乘(Medium)

给定两个以字符串形式表示的非负整数 `num1` 和 `num2`，返回 `num1` 和 `num2` 的乘积，它们的乘积也表示为字符串形式。

**示例 1:**

```
输入: num1 = "2", num2 = "3"
输出: "6"
```

**示例 2:**

```
输入: num1 = "123", num2 = "456"
输出: "56088"
```

**说明：**

1. `num1` 和 `num2` 的长度小于110。
2. `num1` 和 `num2` 只包含数字 `0-9`。
3. `num1` 和 `num2` 均不以零开头，除非是数字 0 本身。
4. **不能使用任何标准库的大数类型（比如 BigInteger）**或**直接将输入转换为整数来处理**。

**解答：**

**思路：**

完全模拟乘法过程。

1. m位的数字乘以n位的数字的结果最大为m+n位：
   - 999*99 < 1000*100 = 100000，最多为3+2 = 5位数。
2. 先将字符串逆序便于从最低位开始计算。

```python
class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        lookup = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
        if num1 == '0' or num2 == '0':
            return '0'
        num1,num2 = num1[::-1],num2[::-1]
        
        tmp_res = [0 for _ in range(len(num1)+len(num2))]
        for i in range(len(num1)):
            for j in range(len(num2)):
                tmp_res[i+j] += lookup[num1[i]] * lookup[num2[j]]
        res = [0 for _ in range(len(num1)+len(num2))]
        for i in range(len(tmp_res)):
            res[i] = tmp_res[i]%10
            if i < len(num1)+len(num2)-1:
                tmp_res[i+1] += tmp_res[i]/10
        return ''.join(str(i) for i in res[::-1]).lstrip('0')
```

## 4. 字符串相加(Easy)(415)

给定两个字符串形式的非负整数 `num1` 和`num2` ，计算它们的和。

**注意：**

1. `num1` 和`num2` 的长度都小于 5100.
2. `num1` 和`num2` 都只包含数字 `0-9`.
3. `num1` 和`num2` 都不包含任何前导零。
4. **你不能使用任何內建 BigInteger 库， 也不能直接将输入的字符串转换为整数形式。**

解答：

思路：

和上述同样的题目

```python
class Solution(object):
    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        num1,num2 = num1[::-1],num2[::-1]
        l1 = len(num1)
        l2 = len(num2)
        if l1 < l2:
            num1,num2 = num2,num1
            l1,l2 = l2,l1
        tmp = 0
        res = []
        for i in range(l1):
            tmp += int(num1[i])
            if i < l2:
                tmp += int(num2[i])
            res.append(tmp%10)
            tmp = tmp/10
        if tmp:
            res.append(tmp)
        return ''.join(str(c) for c in res[::-1])
```



## 5. 通配符匹配(Hard)

给定一个字符串 (`s`) 和一个字符模式 (`p`) ，实现一个支持 `'?'` 和 `'*'` 的通配符匹配。

```
'?' 可以匹配任何单个字符。
'*' 可以匹配任意字符串（包括空字符串）。
```

两个字符串**完全匹配**才算匹配成功。

**说明:**

- `s` 可能为空，且只包含从 `a-z` 的小写字母。
- `p` 可能为空，且只包含从 `a-z` 的小写字母，以及字符 `?` 和 `*`。

**示例 1:**

```
输入:
s = "aa"
p = "a"
输出: false
解释: "a" 无法匹配 "aa" 整个字符串。
```

**示例 2:**

```
输入:
s = "aa"
p = "*"
输出: true
解释: '*' 可以匹配任意字符串。
```

**示例 3:**

```
输入:
s = "cb"
p = "?a"
输出: false
解释: '?' 可以匹配 'c', 但第二个 'a' 无法匹配 'b'。
```

**示例 4:**

```
输入:
s = "adceb"
p = "*a*b"
输出: true
解释: 第一个 '*' 可以匹配空字符串, 第二个 '*' 可以匹配字符串 "dce".
```

**示例 5:**

```
输入:
s = "acdcb"
p = "a*c?b"
输入: false
```

**解答：**

**思路：**

**很显然，第一种思想，动态规划，**

dp\[i\]\[j\]表示s\[:i\]和p\[:j\]是否匹配。

首先dp\[0\]\[0\] = True

其次，dp\[0\]\[j\] = dp\[0\]\[j-1\] if p\[j\] == '*'

其次，dp\[j\]\[0\] = False

动态方程：

```python
if s[i] == p[j] or p[j] == '?' and dp[i-1][j-1]:
		dp[i][j] = True
if p[j] == '*':
  	dp[i][j] = dp[i-1][j] or dp[i][j-1]
```

下面是代码：

```python
class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        dp = [[False] * (len(p)+1) for i in range(len(s)+1)]
        dp[0][0] = True
        for j in range(1,len(p)+1):
            if p[j-1] == '*':
                dp[0][j] = dp[0][j-1]
        for i in range(1,len(s)+1):
            for j in range(1,len(p)+1):
                if p[j-1] == '*':
                    dp[i][j] = dp[i-1][j] or dp[i][j-1]
                elif s[i-1]==p[j-1] or p[j-1] == '?':
                    dp[i][j] = dp[i-1][j-1]
        return dp[-1][-1]
```

**思路二，双指针法**

双指针法还是很难理解的。

具体看代码吧

```python
class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        point_s,point_p = 0,0
        match = 0
        star = -1
        while point_s < len(s):
            if point_p < len(p) and (s[point_s] == p[point_p] or p[point_p] == '?'):
                point_s += 1
                point_p += 1
            elif point_p < len(p) and p[point_p] == '*':
                star = point_p
                match = point_s
                point_p += 1
            elif star != -1:
                point_p = star + 1
                match += 1
                point_s = match
            else:
                return False
        while point_p < len(p) and p[point_p] == '*':
            point_p += 1
        return point_p == len(p)
```



## 6. 跳跃游戏2(Hard)

给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

**示例:**

```
输入: [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
```

**说明:**

假设你总是可以到达数组的最后一个位置。

**解答：**

**思路：**

**贪心规律**

	*  在到达**某点**前，若一直不跳跃，如果发现**从该点无法跳跃更远的地方**了，在这之前，肯定有一次必要的跳跃

* 在无法到达更远的地方时，在这之前应该跳到一个可以到达更远位置的位置

**算法思路**

1. 设置cur 为当前可达的最远位置
2. pre 为遍历各个位置过程中，各个位置能达到的最远位置
3. res 为最少跳跃次数
4. 利用i遍历nums数组，如超过cur 则res加1, cur=pre
5. 遍历过程中，如果pre< nums[i] + i 则更新pre
6. i表示当前位置，num[i]表示当前可以跳多远

```python
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)<2:
            return 0
        cur = nums[0]
        pre = nums[0]
        res = 1
        for i in range(len(nums)):
            if cur < i:
                cur = pre
                res += 1
            pre = max(pre,i + nums[i])
        return res
```

另一份AC代码可能更好理解：

cur_far表示当前下一步可跳最远距离

cur_end表示上一跳所跳的位置

i表示当前所在的位置

```python
class Solution:
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        cur_end, cur_farthest, step = 0, 0, 0
        for i in range(len(nums)-1):
            cur_farthest = max(cur_farthest, i+nums[i])
            if cur_farthest >= len(nums) - 1:
                step += 1
                return step
            if i == cur_end:
                cur_end = cur_farthest
                step += 1
        return step
```

**Very elegant method, but it took me a long time to understand. Some comment for the above:**

**e: longest distance in current minimum step**

**sc: minimum steps for reaching e**

**From i to e, even max is changed in a loop, it is reachable in one step.**



**思路三：动态规划**

超级容易理解。
dp[i]代表的是到达index为i的位置的最少步数, 依然超时。

```python
class Solution:
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums or len(nums) == 0:
            return 0
        dp = [sys.maxsize] * len(nums)
        dp[0] = 0
        for i in range(1, len(nums)):
            for j in range(i):
                if j + nums[j] >= i:
                    dp[i] = min(dp[i], dp[j]+1)
        return dp[-1]
```

