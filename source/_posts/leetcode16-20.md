---

title: leetcode16-20
date: 2019-06-20 15:06:39
tags: leetcode
categories: 算法

---

## 1. 最接近的三数之和(Medium)

给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。

```
例如，给定数组 nums = [-1，2，1，-4], 和 target = 1.

与 target 最接近的三个数的和为 2. (-1 + 2 + 1 = 2).
```

**解答：**

**思路：**

**排序加双指针。很简单。**

1. 标签：排序和双指针
2. 本题目因为要计算三个数，如果靠暴力枚举的话时间复杂度会到 $O(n^3)$，需要降低时间复杂度
3. 首先进行数组排序，时间复杂度 $O(nlogn)$
4. 在数组 nums 中，进行遍历，每遍历一个值利用其下标i，形成一个固定值 nums[i]
5. 再使用前指针指向 start = i + 1 处，后指针指向 end = nums.length - 1 处，也就是结尾处
6. 根据 sum = nums[i] + nums[start] + nums[end] 的结果，判断 sum 与目标 target 的距离，如果更近则更新结果 ans
7. 同时判断 sum 与 target 的大小关系，因为数组有序，如果 sum > target 则 end--，如果 sum < target 则 start++，如果 sum == target 则说明距离为 0 直接返回结果
8. 整个遍历过程，固定值为$ n$ 次，双指针为 $n$ 次，时间复杂度为 $O(n^2)$
9. 总时间复杂度：$O(nlogn) + O(n^2) = O(n^2)$。

```python
class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums.sort()
        n = len(nums)
        res = float('inf')
        for i in range(n):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l = i+1
            r = n-1
            while l < r:
                cur = nums[i] + nums[l] + nums[r]
                if cur == target:
                    return target
                if abs(res - target) > abs(cur - target):
                    res = cur
                if cur > target:
                    r -= 1
                if cur < target:
                    l += 1
        return res
```

**思路二：**

**见提交记录最快的答案。**

```python
class Solution(object):
    def threeSumClosest(self,nums,target):
        nums.sort()
        length = len(nums)
        closest = []
        for i, num in enumerate(nums[0:-2]):
            l, r = i + 1, length - 1
            
            # different with others' solution
            if num + nums[r] + nums[r - 1] < target:
                closest.append(num + nums[r] + nums[r - 1])
            elif num + nums[l] + nums[l + 1] > target:
                closest.append(num + nums[l] + nums[l + 1])
            else:
                while l < r:
                    closest.append(num + nums[l] + nums[r])
                    if num + nums[l] + nums[r] < target:
                        l += 1
                    elif num + nums[l] + nums[r] > target:
                        r -= 1
                    else:
                        return target

        closest.sort(key=lambda x: abs(x - target))
        return closest[0]
```

**执行用时 :32 ms, 在所有 Python 提交中击败了100.00%的用户**

**内存消耗 :11.8 MB, 在所有 Python 提交中击败了14.55%的用户**



## 2. 电话号码中的字母组合(Medium)

给定一个仅包含数字 `2-9` 的字符串，返回所有它能表示的字母组合。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

![](/images/17_telephone_keypad.png)

**示例:**

```
输入："23"
输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
```

**解答：**

**思路：**

**回溯算法**
	回溯是一种通过穷举所有可能情况来找到所有解的算法。如果一个候选解最后被发现并不是可行解，回溯算法会舍弃它，并在前面的一些步骤做出一些修改，并重新尝试找到可行解。

给出如下回溯函数 `backtrack(combination, next_digits)` ，它将一个目前已经产生的组合 combination 和接下来准备要输入的数字 next_digits 作为参数。

如果没有更多的数字需要被输入，那意味着当前的组合已经产生好了。 如果还有数字需要被输入：遍历下一个数字所对应的所有映射的字母。将当前的字母添加到组合最后，也就是 `combination = combination + letter` 。 重复这个过程，输入剩下的数字： `backtrack(combination + letter, next_digits[1:])` 。

```python
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        phone = {'2': ['a', 'b', 'c'],
                 '3': ['d', 'e', 'f'],
                 '4': ['g', 'h', 'i'],
                 '5': ['j', 'k', 'l'],
                 '6': ['m', 'n', 'o'],
                 '7': ['p', 'q', 'r', 's'],
                 '8': ['t', 'u', 'v'],
                 '9': ['w', 'x', 'y', 'z']}
        def backtrack(combination,next_digits):
            if len(next_digits) == 0:
                output.append(combination)
            else:
                for char in phone[next_digits[0]]:
                    backtrack(combination+char,next_digits[1:])
        output = []
        if not digits or len(digits) == 0:
            return output
        backtrack('',digits)
        return output
```

**思路二：**

每次更新一个字母，类似**BFS**

```python
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        num2char = {'2':['a','b','c'],
                   '3':['d','e','f'],
                   '4':['g','h','i'],
                   '5':['j','k','l'],
                   '6':['m','n','o'],
                   '7':['p','q','r','s'],
                   '8':['t','u','v'],
                   '9':['w','x','y','z']}    
        if not digits:
            return []
        res = [""]
        for num in digits:
            next_res = []
            for alp in num2char[num]:
                for tmp in res:
                    next_res.append(tmp + alp)
            res = next_res
        return res
```

执行用时 :16 ms, 在所有 Python 提交中击败了99.12%的用户



## 3. 四数之和(Medium)

给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。

注意：答案中不可以包含重复的四元组。

**示例：**

```
给定数组 nums = [1, 0, -1, 0, -2, 2]，和 target = 0。

满足要求的四元组集合为：
[
  [-1,  0, 0, 1],
  [-2, -1, 1, 2],
  [-2,  0, 0, 2]
]
```

**解答：**

**思路一：时间复杂度$O(n^3)$，空间复杂度$O(1)$**

使用3sum改，固定两个数，活动别的

```python
class Solution:
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        n = len(nums)
        nums.sort()
        res = []
        for i in range(n):
            for j in range(i + 1, n):
                l, r = j + 1, n - 1
                while l < r:
                    temp = nums[i] + nums[j] + nums[l] + nums[r]
                    if temp == target:
                        if [nums[i], nums[j], nums[l], nums[r]] not in res:
                            res.append([nums[i], nums[j], nums[l], nums[r]])
                        l += 1
                        r -= 1
                    elif temp > target:
                        r -= 1
                    else:
                        l += 1
        return res
```

**思路：时间复杂度$O(n^3)$，空间复杂度$O(1)​**$

使用双循环固定两个数，用双指针找另外两个数，通过比较与target 的大小，移动指针。里面有一些优化，可以直接看代码，很好理解！所以时间复杂度不超过$O(n^3)$

```python
class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        n = len(nums)
        if n < 4:
            return []
        nums.sort()
        res = []
        for i in range(n-3):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            if nums[i] + nums[i+1] + nums[i+2] + nums[i+3] > target:
                break
            if nums[i] + nums[n-1] + nums[n-2] + nums[n-2] < target:
                continue
            for j in range(i+1,n-2):
                if j - i > 1 and nums[j] == nums[j-1]:
                    continue
                if nums[i] + nums[j] + nums[j+1] + nums[j+2] > target:
                    break
                if nums[i] + nums[j] + nums[n-1] + nums[n-2] < target:
                    continue
                l = j+1
                r = n-1
                while l < r:
                    tmp = nums[i] + nums[j] + nums[l] + nums[r]
                    if tmp == target:
                        res.append([nums[i],nums[j],nums[l],nums[r]])
                        while l < r and nums[l] == nums[l+1]:
                            l += 1
                        while l < r and nums[r] == nums[r-1]:
                            r -= 1
                    if tmp > target:
                        r -= 1
                    else:
                        l += 1
        return res
```

## 4. 删除链表中倒数第N个节点(Medium)

给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。

**示例：**

```
给定一个链表: 1->2->3->4->5, 和 n = 2.

当删除了倒数第二个节点后，链表变为 1->2->3->5.
```

**说明：**给定的 n 保证是有效的。

解答：

思路一：核心词，**dummy head,快慢指针，双指针。**

1. 标签：链表 
2. 整体思路是让前面的指针先移动n步，之后前后指针共同移动直到前面的指针到尾部为止
3. 首先设立预先指针 pre，预先指针是一个小技巧：**对于链表问题，返回结果为头结点时，通常需要先初始化一个预先指针 pre，该指针的下一个节点指向真正的头结点head。使用预先指针的目的在于链表初始化时无可用节点值，而且链表构造过程需要指针移动，进而会导致头指针丢失，无法返回结果。**
4. 设预先指针 pre 的下一个节点指向 head，设前指针为 start，后指针为 end，二者都等于 pre
5. start 先向前移动n步
6. 之后 start 和 end 共同向前移动，此时二者的距离为 n，当 start 到尾部时，end 的位置恰好为倒数第 n 个节点
7. 因为要删除该节点，所以要移动到该节点的前一个才能删除，所以循环结束条件为 start.next != null
8. 删除后返回 pre.next，为什么不直接返回 head 呢，因为 head 有可能是被删掉的点
9. 时间复杂度：$O(n)$

切记最后要返回`dummy.next`而不是`head`，因为可能删的就是head，例如：

输入链表为`[1]`, `n = 1`, 应该返回`None`而不是`[1]`

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        slow = fast = dummy = ListNode(-1)
        dummy.next = head
        for i in range(n):
            fast = fast.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next   
```

### Follow Up

**Could you do this in one pass?**

**思路：把for loop放到while 里面来实现**

```python
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        slow = fast = dummy = ListNode(-1)
        dummy.next = head
        count = 0
        while fast.next:
            if count < n:
                count += 1
                fast= fast.next
            else:
                fast = fast.next
                slow = slow.next
        slow.next = slow.next.next
        return dummy.next
```



## 5. 有效的括号(Easy)

给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。

有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。

注意空字符串可被认为是有效字符串。

**示例 1:**

```
输入: "()"
输出: true
```


**示例 2:**

```
输入: "()[]{}"
输出: true
```


**示例 3:**

```
输入: "(]"
输出: false
```


**示例 4:**

```
输入: "([)]"
输出: false
```


**示例 5:**

```
输入: "{[]}"
输出: true
```

**解答：**

**思路一：利用栈,时间复杂度$O(n)$，空间复杂度$O(n)$**

**算法**

1. 初始化栈 S。
2. 一次处理表达式的每个括号。
3. 如果遇到开括号，我们只需将其推到栈上即可。这意味着我们将稍后处理它，让我们简单地转到前面的 子表达式。
4. 如果我们遇到一个闭括号，那么我们检查栈顶的元素。如果栈顶的元素是一个 相同类型的 左括号，那么我们将它从栈中弹出并继续处理。否则，这意味着表达式无效。
5. 如果到最后我们剩下的栈中仍然有元素，那么这意味着表达式无效。

```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        mapping = {')':'(','}':'{',']':'['}
        stack = []
        for char in s:
            if char in mapping:
                top_ele = stack.pop() if stack else '#'
                if top_ele != mapping[char]:
                    return False                                
            else:
                stack.append(char)
        return not stack
```

**思路二：时间复杂度$O(n)$，空间复杂度$O(n)$**

因为一共只有三种状况"(" -> ")", "[" -> "]", "{" -> "}".

一遇到左括号就入栈，右括号出栈，这样来寻找对应

需要检查几件事：

- 出现右括号时stack里还有没有东西
- 出stack时是否对应
- 最终stack是否为空

```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        leftP = '([{'
        rightP = ')]}'
        stack = []
        for char in s:
            if char in leftP:
                stack.append(char)
            if char in rightP:
                if not stack:
                    return False
                tmp = stack.pop()
                if char == ')' and tmp != '(':
                    return False
                if char == ']' and tmp != '[':
                    return False       
                if char == '}' and tmp != '{':
                    return False
        return stack == []
```

