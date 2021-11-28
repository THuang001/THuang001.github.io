---
title: 字符串算法题集合
date: 2019-07-10 15:07:29
tags: string
categories: 算法
---

# 一、综述

# 二、字符串类题目思路

# 三、例题

## 1. 公共前缀问题

最长公共前缀

### 二分查找实现

基本原理同暴力实现，只是最初的比较对象，由基准元素的一个一个比较，变为基准元素的前一半进行比较，**这里实现选取的基准元素改为数组中的最短元素。**

**注意：**

1. 左指针还是右指针移动的标记的设置，即实现中的flag变量
2. 遍历结束，mid的值就是元素minElement中最长前缀的停止位置（不包含mid所在位置）

```python
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs)<1:           
            return ""
        if len(strs) ==1:        
            return strs[0]
        
        minElement = strs[0]
        for i in range(len(strs)):
            if len(strs[i])<len(minElement):
                minElement = strs[i]
        left = 0
        right =len(minElement)
        mid = (left+right)//2
        while left<right:
            flag = True           
            for j in range(len(strs)):
                if  minElement[:mid+1] != strs[j][:mid+1]: 
                    right = mid
                    flag = False
                    break
            if flag :
                left = mid+1
            mid = (left+right)//2   
            
        return minElement[:mid]
```

### ZIP()大法

```python
class Solution:
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        res = ""
        for tmp in zip(*strs):
            tmp_set = set(tmp)
            if len(tmp_set) == 1:
                res += tmp[0]
            else:
                break
        return res
```



## 2. 字符串匹配的KMP算法，BM算法，Sunday算法

### KMP算法

这个也很容易理解。

一个辅助数组next[]表示当前字符前面的字符串的最长回文的初始位置。

然后每一次匹配的时候，当遇到匹配不成功时，主字符串移动next[]个位置。

#### **KMP的算法流程**

假设现在文本串S匹配到 i 位置，模式串P匹配到 j 位置

1. 如果j = -1，或者当前字符匹配成功（即S[i] == P[j]），都令i++，j++，继续匹配下一个字符；

2. 如果j != -1，且当前字符匹配失败（即S[i] != P[j]），则令 i 不变，j = next[j]。此举意味着失配时，模式串P相对于文本串S向右移动了j - next [j] 位。

#### 递推计算next 数组

next 数组各值的含义：**代表当前字符之前的字符串中，有多大长度的相同前缀后缀**。例如如果next [j] = k，代表j 之前的字符串中有最大长度为*k* 的相同前缀后缀。

1. 如果对于值k，已有p0 p1, ..., pk-1 = pj-k pj-k+1, ..., pj-1，相当于next[j] = k。

2. 对于P的前j+1个序列字符：

   若p[k] == p[j]，则next[j + 1 ] = next [j] + 1 = k + 1；

   若p[k ] ≠ p[j]，如果此时p[ next[k] ] == p[j ]，则next[ j + 1 ] =  next[k] + 1，否则继续递归前缀索引k = next[k]，而后重复此过程。 相当于在字符p[j+1]之前不存在长度为k+1的前缀"p0 p1, …, pk-1 pk"跟后缀“pj-k pj-k+1, …, pj-1 pj"相等，那么是否可能存在另一个值t+1 < k+1，使得长度更小的前缀 “p0 p1, …, pt-1 pt” 等于长度更小的后缀 “pj-t pj-t+1, …, pj-1 pj” 呢？如果存在，那么这个t+1 便是next[ j+1]的值，此相当于利用已经求得的next 数组（next [0, ..., k, ..., j]）进行P串前缀跟P串后缀的匹配。

---------------------
```python
def getNext(p):
    """
    p为模式串
    返回next数组，即部分匹配表
    等同于从模式字符串的第1位(注意，不包括第0位)开始对自身进行匹配运算。
    """
    nex = [0] * len(p)
    nex[0] = -1
    i = 0
    j = -1
    while i < len(p) - 1:   # len(p)-1防止越界，因为nex前面插入了-1
        if j == -1 or p[i] == p[j]:
            i += 1
            j += 1
            nex[i] = j     # 这是最大的不同：记录next[i]
        else:
            j = nex[j]    
    return nex
```

---

```python
def KMP(s, p):
    """
    s为主串
    p为模式串
    如果t里有p，返回打头下标
    """
    nex = getNext(p)
    i = j = 0   # 分别是s和p的指针
    while i < len(s) and j < len(p):
        if j == -1 or s[i] == p[j]: # j==-1是由于j=next[j]产生
            i += 1
            j += 1
        else:
            j = nex[j]
            
    if j == len(p): # 匹配到了
        return i - j
    else:
        return -1
```

## 3. 字符串的排列、组合





## 4. 字符串的回文系列



## 5. 字符串的翻转，旋转，替换等



## 6. 