---

title: leetcode86-100
date: 2019-06-27 22:58:14
tags: leetcode
categories: 算法

---

## 1. 分隔链表(Medium)

给定一个链表和一个特定值 *x*，对链表进行分隔，使得所有小于 *x* 的节点都在大于或等于 *x*的节点之前。

你应当保留两个分区中每个节点的初始相对位置。

**示例:**

```
输入: head = 1->4->3->2->5->2, x = 3
输出: 1->2->2->4->3->5
```

**解答：**

**思路：双指针，dummy大法**

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        p1 = dummy1 = ListNode(-1)
        p2 = dummy2 = ListNode(-1)
        while head:
            if head.val < x:
                p1.next = ListNode(head.val)
                p1 = p1.next
            else:
                p2.next = ListNode(head.val)
                p2 = p2.next
            head = head.next
        p2.next = None
        p1.next = dummy2.next
        return dummy1.next
```



## 2. 扰乱字符串(Hard)

给定一个字符串 *s1*，我们可以把它递归地分割成两个非空子字符串，从而将其表示为二叉树。

下图是字符串 *s1* = `"great"` 的一种可能的表示形式。

```
    great
   /    \
  gr    eat
 / \    /  \
g   r  e   at
           / \
          a   t
```

在扰乱这个字符串的过程中，我们可以挑选任何一个非叶节点，然后交换它的两个子节点。

例如，如果我们挑选非叶节点 `"gr"` ，交换它的两个子节点，将会产生扰乱字符串 `"rgeat"` 。

```
    rgeat
   /    \
  rg    eat
 / \    /  \
r   g  e   at
           / \
          a   t
```

我们将 `"rgeat”` 称作 `"great"` 的一个扰乱字符串。

同样地，如果我们继续将其节点 `"eat"` 和 `"at"` 进行交换，将会产生另一个新的扰乱字符串 `"rgtae"` 。

```
    rgtae
   /    \
  rg    tae
 / \    /  \
r   g  ta  e
       / \
      t   a
```

我们将 `"rgtae”` 称作 `"great"` 的一个扰乱字符串。

给出两个长度相等的字符串 *s1* 和 *s2*，判断 *s2* 是否是 *s1* 的扰乱字符串。

**示例 1:**

```
输入: s1 = "great", s2 = "rgeat"
输出: true
```

**示例 2:**

```
输入: s1 = "abcde", s2 = "caebd"
输出: false
```

**解答：**

**思路：递归法，很好理解**

```python
class Solution(object):
    def isScramble(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        if len(s1) != len(s2) or sorted(s1) != sorted(s2):
            return False
        if len(s1) < 4 or s1 == s2:
            return True
        for i in range(1,len(s1)):
            if self.isScramble(s1[:i],s2[:i]) and self.isScramble(s1[i:],s2[i:]):
                return True
            if self.isScramble(s1[:i],s2[-i:]) and self.isScramble(s1[i:],s2[:-i]):
                return True
        return False
```



## 3. 合并两个有序数组(Easy)

给定两个有序整数数组 *nums1* 和 *nums2*，将 *nums2* 合并到 *nums1* 中*，*使得 *num1* 成为一个有序数组。

**说明:**

- 初始化 *nums1* 和 *nums2* 的元素数量分别为 *m* 和 *n*。
- 你可以假设 *nums1* 有足够的空间（空间大小大于或等于 *m + n*）来保存 *nums2*中的元素。

**示例:**

```
输入:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

输出: [1,2,2,3,5,6]
```

解答：

思路：双指针，很简单

合并两个有序数组，从后往前依次读取存储到数组中，减少了空间复杂度

```python
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        p1 = m-1
        p2 = n-1
        p = m+n-1
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]
                p1 -= 1
            else:
                nums1[p] = nums2[p2]
                p2 -= 1
            p -= 1
        nums1[:p2+1] = nums2[:p2+1]
```



## 4. 格雷编码(Medium)

格雷编码是一个二进制数字系统，在该系统中，两个连续的数值仅有一个位数的差异。

给定一个代表编码总位数的非负整数 *n*，打印其格雷编码序列。格雷编码序列必须以 0 开头。

**示例 1:**

```
输入: 2
输出: [0,1,3,2]
解释:
00 - 0
01 - 1
11 - 3
10 - 2

对于给定的 n，其格雷编码序列并不唯一。
例如，[0,2,3,1] 也是一个有效的格雷编码序列。

00 - 0
10 - 2
11 - 3
01 - 1
```

**示例 2:**

```
输入: 0
输出: [0]
解释: 我们定义格雷编码序列必须以 0 开头。
     给定编码总位数为 n 的格雷编码序列，其长度为 2n。当 n = 0 时，长度为 20 = 1。
     因此，当 n = 0 时，其格雷编码序列为 [0]。
```

**解答：**

**思路：格雷码的计算公式**

![image.png](https://pic.leetcode-cn.com/1013850d7f6c8cf1d99dc0ac3292264b74f6a52d84e0215f540c80952e184f41-image.png)

```python
class Solution(object):
    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        res = []
        for i in range(2**n):
            res.append((i>>1)^i)
        return res
```



## 5. 子集2(Medium)

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

**解答：**

**思路：回溯算法，之后会建一个回溯算法专题专门讲回溯系列。**

```python
class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        nums.sort()
        def helper(idx,tmp):
            res.append(tmp)
            for i in range(idx,len(nums)):
                if i > idx and nums[i] == nums[i-1]:
                    continue
                helper(i+1,tmp+[nums[i]])
        helper(0,[])
        return res
```



## 6. 解码方法(Medium)

一条包含字母 `A-Z` 的消息通过以下方式进行了编码：

```
'A' -> 1
'B' -> 2
...
'Z' -> 26
```

给定一个只包含数字的**非空**字符串，请计算解码方法的总数。

**示例 1:**

```
输入: "12"
输出: 2
解释: 它可以解码为 "AB"（1 2）或者 "L"（12）。
```

**示例 2:**

```
输入: "226"
输出: 3
解释: 它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。
```

**解答：**

**思路：动态规划：**

**$dp(i) = dp(i-1) + dp(i-2)$**

```python
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s or len(s) == 0:
            return 0
        dp = [0] * len(s)
        if s[0] == '0':
            return 0
        else:
            dp[0] = 1
        if len(s) == 1:
            return dp[-1]
        if s[1] !=  '0':
            dp[1] += 1
        if 10 <= int(s[:2]) <= 26:
            dp[1] += 1
        for i in range(2,len(s)):
            if s[i]+s[i-1] == '00':
                return 0
            if s[i] != '0':
                dp[i] += dp[i-1]
            if 10 <= int(s[i-1:i+1]) <= 26:
                dp[i] += dp[i-2]
        return dp[-1]
```



## 7. 反转链表2(Medium)

反转从位置 *m* 到 *n* 的链表。请使用一趟扫描完成反转。

**说明:**
1 ≤ *m* ≤ *n* ≤ 链表长度。

**示例:**

```
输入: 1->2->3->4->5->NULL, m = 2, n = 4
输出: 1->4->3->2->5->NULL
```

解答：

思路：题解区一个解题思路，很简单。

对于链表的问题，根据以往的经验一般都是要建一个dummy node，连上原链表的头结点，这样的话就算头结点变动了，我们还可以通过dummy->next来获得新链表的头结点。这道题的要求是只通过一次遍历完成，就拿题目中的例子来说，变换的是2,3,4这三个点，我们需要找到第一个开始变换结点的前一个结点，只要让pre向后走m-1步即可，为啥要减1呢，因为题目中是从1开始计数的，这里只走了1步，就是结点1，用pre指向它。万一是结点1开始变换的怎么办，这就是我们为啥要用dummy结点了，pre也可以指向dummy结点。然后就要开始交换了，由于一次只能交换两个结点，所以我们按如下的交换顺序：

1 -> 2 -> 3 -> 4 -> 5 -> NULL

1 -> 3 -> 2 -> 4 -> 5 -> NULL

1 -> 4 -> 3 -> 2 -> 5 -> NULL

我们可以看出来，总共需要n-m步即可，第一步是将结点3放到结点1的后面，第二步将结点4放到结点1的后面。这是很有规律的操作，那么我们就说一个就行了，比如刚开始，pre指向结点1，cur指向结点2，然后我们建立一个临时的结点t，指向结点3（注意我们用临时变量保存某个结点就是为了首先断开该结点和前面结点之间的联系，这可以当作一个规律记下来），然后我们断开结点2和结点3，将结点2的next连到结点4上，也就是 cur->next = t->next，再把结点3连到结点1的后面结点（即结点2）的前面，即 t->next = pre->next，最后再将原来的结点1和结点2的连接断开，将结点1连到结点3，即 pre->next = t。这样我们就完成了将结点3取出，加入结点1的后方。第二步将结点4取出，加入结点1的后方，也是同样的操作。

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        if not head or m == n:
            return head
        pre = dummy = ListNode(None)
        dummy.next = head
        for i in range(m-1):
            pre = pre.next
        cur = pre.next
        for i in range(n-m):
            tmp = cur.next
            cur.next = tmp.next
            tmp.next = pre.next
            pre.next = tmp
        return dummy.next
```



## 8. 复原IP地址(Medium)

给定一个只包含数字的字符串，复原它并返回所有可能的 IP 地址格式。

**示例:**

```
输入: "25525511135"
输出: ["255.255.11.135", "255.255.111.35"]
```

**解答：**

**思路：回溯算法**

在回溯算法专题中详细讲解。

```python
class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        res = []
        n = len(s)
        def backtrack(i,tmp,flag):
            if i == n and flag == 0:
                res.append(tmp[:-1])
                return 
            if flag < 0:
                return
            for j in range(i, i+3):
                if j < n:
                    if i == j and s[j] == '0':
                        backtrack(j+1,tmp+s[j]+'.',flag-1)
                        break
                    if 0 < int(s[i:j+1]) <= 255:
                        backtrack(j+1,tmp+s[i:j+1]+'.',flag-1)
        backtrack(0,'',4)
        return res
```



## 9. 二叉树的中序遍历(Medium)

给定一个二叉树，返回它的*中序* 遍历。

**示例:**

```
输入: [1,null,2,3]
   1
    \
     2
    /
   3

输出: [1,3,2]
```

**进阶:** 递归算法很简单，你可以通过迭代算法完成吗？

**解答：**

**思路一：递归**

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        res = []
        if root.left:
            res += self.inorderTraversal(root.left)
        res.append(root.val)
        if root.right:
            res += self.inorderTraversal(root.right)
        return res
```



**思路二：迭代**

用栈，这都是熟悉的不能再熟悉了。

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        res = []
        stack = []
        p = root
        while p or len(stack):
            while p:
                stack.append(p)
                p = p.left
            if stack:
                p = stack.pop()
                res.append(p.val)
                p = p.right
        return res
```

同时，附上如何进行前序遍历和后续遍历

前序遍历的两种方式：

递归：

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        res = []
        res.append(root.val)
        res += self.preorderTraversal(root.left)
        res += self.preorderTraversal(root.right)
        return res
```

迭代：

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        res = []
        stack = [root]
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res
```

迭代方法二：和中序遍历类似：

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        res = []
        node = root
        stack = []
        while node or len(stack)>0:
            while node:
                stack.append(node)
                res.append(node.val)
                node = node.left
            if len(stack)>0:
                node = stack.pop()
                node = node.right
        return res
```

后序遍历：**

**递归：**

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        if not root:
            return res
        res += self.postorderTraversal(root.left)
        res += self.postorderTraversal(root.right)
        res.append(root.val)
        return res
```

**迭代：**

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        if not root:
            return res
        stack = [root]
        while stack:
            node = stack.pop()
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
            res.append(node.val)
        return res[::-1]
```



## 10. 不同的搜索二叉树2(Medium)

给定一个整数 *n*，生成所有由 1 ... *n* 为节点所组成的**二叉搜索树**。

**示例:**

```
输入: 3
输出:
[
  [1,null,3,2],
  [3,2,null,1],
  [3,1,null,null,2],
  [2,1,3],
  [1,null,2,null,3]
]
解释:
以上的输出对应以下 5 种不同结构的二叉搜索树：

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
```

**解答：**

**思路：递归调用，有意思。**

因为知道这是一棵二叉搜索树，所以**left.val < root.val < right.val**

然后可以任意取一个node作为root，递归调用左边用返回的node作为left，递归调用右边用返回的node作为right

注意考虑n为0的情况，应该返回[]而不是[[]]

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        if n == 0:
            return []
        
        def helper(nums):
            if not nums:
                return [None]
            res = []
            for i in range(len(nums)):
                for l in helper(nums[:i]):
                    for r in helper(nums[i+1:]):
                        node = TreeNode(nums[i])
                        node.left = l
                        node.right = r
                        res.append(node)
            return res
        return helper(list(range(1,n+1)))
```





## 11. 不同的二叉搜索树(Medium)

​	给定一个整数 *n*，求以 1 ... *n* 为节点组成的二叉搜索树有多少种？

**示例:**

```
输入: 3
输出: 5
解释:
给定 n = 3, 一共有 5 种不同结构的二叉搜索树:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
```

**解答：**

**思路：动态规划**
假设n个节点存在二叉排序树的个数是$G(n)$，令$f(i)$为以$i$为根的二叉搜索树的个数，则
$G(n) = f(1) + f(2) + f(3) + f(4) + ... + f(n)$

当$i$为根节点时，其左子树节点个数为$i-1$个，右子树节点为$n-i$，则
$f(i) = G(i-1)*G(n-i)$

综合两个公式可以得到 **卡特兰数** 公式
$G(n) = G(0)*G(n-1)+G(1)*(n-2)+...+G(n-1)*G(0)$

```python
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [0] * (n+1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2,n+1):
            for j in range(i):
                dp[i] += dp[j] * dp[i-j-1]
        return dp[-1]
```



## 12. 交错字符串(Hard)

给定三个字符串 *s1*, *s2*, *s3*, 验证 *s3* 是否是由 *s1* 和 *s2* 交错组成的。

**示例 1:**

```
输入: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
输出: true
```

**示例 2:**

```
输入: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
输出: false
```

**解答：**

**思路：动态规划**

动态方程:

用$dp[i][j]$表示s1的前i元素和s2前j元素是否交错组成s3前i+j元素

所以有动态方程:

$dp[i][j] = (dp[i-1][j] \&\& s3[i+j-1] == s1[i-1]) || (dp[i][j-1] \&\& s2[j-1] == s3[i+j-1])$

注意:针对第一行,第一列要单独考虑

```python
class Solution(object):
    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        if len(s1) +len(s2) != len(s3):
            return False
        dp = [[False] * (len(s2)+1) for _ in range(len(s1)+1)]
        dp[0][0] = True
        for i in range(1,len(s1)+1):
            dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
        for j in range(1,len(s2)+1):
            dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
        for i in range(1,len(s1)+1):
            for j in range(1,len(s2)+1):
                dp[i][j] = (dp[i-1][j] and s1[i-1]== s3[i+j-1]) or (dp[i][j-1] and s2[j-1]==s3[i+j-1])
        return dp[-1][-1]
```



## 13. 验证二叉搜索树(Medium)

给定一个二叉树，判断其是否是一个有效的二叉搜索树。

假设一个二叉搜索树具有如下特征：

- 节点的左子树只包含**小于**当前节点的数。
- 节点的右子树只包含**大于**当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

**示例 1:**

```
输入:
    2
   / \
  1   3
输出: true
```

**示例 2:**

```
输入:
    5
   / \
  1   4
     / \
    3   6
输出: false
解释: 输入为: [5,1,4,null,null,3,6]。
     根节点的值为 5 ，但是其右子节点值为 4 。
```

**解答：**

**思路一：最大最小值**



```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def isBST(root,min_val,max_val):
            if not root:
                return True
            if root.val >= max_val or root.val <= min_val:
                return False
            return isBST(root.left,min_val,root.val) and isBST(root.right,root.val,max_val)
        return isBST(root,float('-inf'),float('inf'))
```

思路二：利用中序遍历比较大小

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        stack = []
        p = root
        pre = None
        while p or stack:
            while p:
                stack.append(p)
                p = p.left
            if stack:
                p = stack.pop()
                if pre and p.val <= pre.val:
                    return False
                pre = p
                p = p.right
        return True
```



## 14. 恢复二叉搜索树(Hard)

二叉搜索树中的两个节点被错误地交换。

请在不改变其结构的情况下，恢复这棵树。

**示例 1:**

```
输入: [1,3,null,null,2]

   1
  /
 3
  \
   2

输出: [3,1,null,null,2]

   3
  /
 1
  \
   2
```

**示例 2:**

```
输入: [3,1,4,null,null,2]

  3
 / \
1   4
   /
  2

输出: [2,1,4,null,null,3]

  2
 / \
1   4
   /
  3
```

**进阶:**

- 使用 O(*n*) 空间复杂度的解法很容易实现。
- 你能想出一个只使用常数空间的解决方案吗？

**解答：**

**思路：**

这道题难点,是找到那两个交换节点,把它交换过来就行了.

这里我们二叉树搜索树的中序遍历(中序遍历遍历元素是递增的)

如下图所示, 中序遍历顺序是 4,2,3,1,我们只要找到节点4和节点1交换顺序即可!

![1561339663404.png](https://pic.leetcode-cn.com/b4ee2fcef6391b0756e5d0e84540b448b9ac122a0efabc9ff505e4fdcdd3bc3c-1561339663404.png)

这里我们有个规律发现这两个节点:

第一个节点,是第一个按照中序遍历时候前一个节点大于后一个节点,我们选取前一个节点,这里指节点4;

第二个节点,是在第一个节点找到之后, 后面出现前一个节点大于后一个节点,我们选择后一个节点,这里指节点1;

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def recoverTree(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        first = second = None
        pre = None
        stack = []
        p = root
        while p or stack:
            while p:
                stack.append(p)
                p = p.left
            if stack:
                p = stack.pop()
            if not first and  pre and pre.val > p.val:
                first = pre
            if first and  pre and pre.val > p.val:
                second = p
            pre = p
            p = p.right
        first.val,second.val = second.val,first.val
```



## 15. 相同的树(Easy)

给定两个二叉树，编写一个函数来检验它们是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

**示例 1:**

```
输入:       1         1
          / \       / \
         2   3     2   3

        [1,2,3],   [1,2,3]

输出: true
```

**示例 2:**

```
输入:      1          1
          /           \
         2             2

        [1,2],     [1,null,2]

输出: false
```

**示例 3:**

```
输入:       1         1
          / \       / \
         2   1     1   2

        [1,2,1],   [1,1,2]

输出: false
```

**解答：**

**思路：二叉树算法的基本思路：明确一个节点要做的事情，然后剩下的事抛给框架。**

```c++
void traverse(TreeNode root) {
    // root 需要做什么？在这做。
    // 其他的不用 root 操心，抛给框架
    traverse(root.left);
    traverse(root.right);
}
```

举两个简单的例子体会一下这个思路。

**如何把二叉树所有的节点中的值加一？**

```
void plusOne(TreeNode root) {
    if (root == null) return;
    		root.val += 1;
		plusOne(root.left);
		plusOne(root.right);
}
```

本题便是如此：

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if not p and not q:
            return True
        if p and q and p.val == q.val:
            return self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)
        return False
```

