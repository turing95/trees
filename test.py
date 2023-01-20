'''
A string of brackets is correctly matched if you can pair every opening bracket up with a later closing bracket, and vice versa. For example, (()()) is correctly matched, and (() and )( are not.

Implement a function which takes a string of brackets and returns the minimum number of brackets you'd have to add to the string to make it correctly matched.

For example, (() could be correctly matched by adding a single closing bracket at the end, so you'd return 1. )( can be correctly matched by adding an opening bracket at the start and a closing bracket at the end, so you'd return 2.

If your string is already correctly matched, you can just return 0.
'''


def count_bracket_additions(string):
    count = 0
    stack = []
    for char in string:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                count += 1
            elif stack[-1] == '(':
                stack.pop()
            else:
                count += 2
    return count + len(stack)


'''
The input nums is supposed to be an array of unique integers ranging from 1 to nums.length (inclusive). However, there is a mistake: one of the numbers in the array is duplicated, which means another number is missing.

Find and return the sum of the duplicated number and the missing number.

Example: in the array [4, 3, 3, 1], 3 is present twice and 2 is missing, so 3 + 2 = 5 should be returned.'''


def find_duplicate_and_missing(nums):
    n = len(nums)
    expected_sum = (n * (n + 1)) // 2
    actual_sum = sum(nums)
    duplicate_and_missing_sum = actual_sum - expected_sum
    for i in range(n):
        if nums[abs(nums[i]) - 1] < 0:
            duplicate = abs(nums[i])
        else:
            nums[abs(nums[i]) - 1] = -nums[abs(nums[i]) - 1]

    missing = next((i + 1) for i, x in enumerate(nums) if x > 0)

    return duplicate + missing



'''

The deletion distance between two strings is the minimum sum of ASCII values of characters that you need to delete in the two strings in order to have the same string. The deletion distance between cat and at is 99, because you can just delete the first character of cat and the ASCII value of 'c' is 99. The deletion distance between cat and bat is 98 + 99, because you need to delete the first character of both words. Of course, the deletion distance between two strings can't be greater than the sum of their total ASCII values, because you can always just delete both of the strings entirely.

Implement an efficient function to find the deletion distance between two strings.

You can refer to the Wikipedia article on the algorithm for edit distance if you want to. The algorithm there is not quite the same as the algorithm required here, but it's similar.'''


def deletion_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                dp[i][j] = sum(ord(c) for c in str2[:j])
            elif j == 0:
                dp[i][j] = sum(ord(c) for c in str1[:i])
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j] + ord(str1[i-1]), dp[i][j-1] + ord(str2[j-1]))
    return dp[m][n]