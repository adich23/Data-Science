# A. Data-structures

## 1. Heap
### Min-heap (Default)
```python
* heap = [[4,'abc'],[2,'def']] 
* heapq.heapify(heap)
* heapq.heappush(heap, item)
* heapq.heappop(heap)

* heapq.heappushpop(heap, item)
```
### Max-heap
```python
insert with **negative values**
* heap = [[-4,'abc'],[-2,'def']] 
```
## 2. String 
### Useful Funtions

## 3. Trie

## 4. Binary-Lifting Trie, LCA

## 5. Segment Tree ,Range-Sum Query
```python
class Node:
    
    def __init__(self, val=0):
        self.val = val
        
class NumArray:
    # Segment Tree
    def __init__(self, nums: List[int]):
        self.nums = nums
        n = len(nums)
        if n==0:
            return
        self.tree = [Node() for i in range(4*n)]
        self.__build(0,0,n-1)
        # for node in self.tree:
        #     print(node.val,end=" ")
        
    def __build(self, n_idx, start, end):
        
        tree = self.tree
        nums = self.nums
        # Leaf node
        if start == end:
            tree[n_idx].val = nums[start]
        else:
            mid = (start+end)//2
            #Left - Right Children
            self.__build(2*n_idx+1, start, mid)
            self.__build(2*n_idx+2, mid+1, end)
        
            tree[n_idx].val = tree[2*n_idx+1].val + tree[2*n_idx+2].val
    
    def __update(self, n_idx, idx, start, end, val):
        
        tree = self.tree
        nums = self.nums
        
        if start == end:
            tree[n_idx].val = val
            nums[idx] =val
        else:
            
            mid = (start+end)//2
            
            # If idx is in the left child, recurse on the left child
            if  start <= idx <= mid:
                self.__update(2*n_idx+1, idx, start, mid, val)       
            else:
            # Right Child
                self.__update(2*n_idx+2, idx, mid+1, end, val)

            tree[n_idx].val = tree[2*n_idx+1].val + tree[2*n_idx+2].val
            
        
        
    def __query(self, n_idx, start, end, l, r):
        
        tree = self.tree
        nums = self.nums    
        
        # range represented by a node is completely outside the given range
        if r < start or end < l:
            return 0
        
        if l <= start and end <= r:
            return tree[n_idx].val
        
        # Partial range
        mid = (start+end)//2
        
        p1 = self.__query(2*n_idx+1, start, mid, l, r)
        p2 = self.__query(2*n_idx+2, mid+1, end, l, r)
        
        return p1+p2

    def update(self, i: int, val: int) -> None:
        
        self.__update(0,i,0,len(self.nums)-1, val )
        
    def sumRange(self, i: int, j: int) -> int:
        return self.__query(0, 0, len(self.nums)-1, i,j)


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# obj.update(i,val)
# param_2 = obj.sumRange(i,j)
# Reference - https://www.hackerearth.com/practice/data-structures/advanced-data-structures/segment-trees/tutorial/

```

## 6. Union Find (Disjoint Set Union)
```python
class DSU:
    
    def __init__(self, n):
        self.par = [x for x in range(n)]
        self.rnk = [0]*n
    
    def find(self, x):
        if self.par[x]!= x:
            self.par[x] = self.find(self.par[x])
        
        return self.par[x]
    
    def union(self, x, y):
        
        xp, yp = self.find(x), self.find(y)
        # rank
        if self.rnk[xp] > self.rnk[yp]:
            self.par[yp] = xp
        elif self.rnk[yp] > self.rnk[xp]:
            self.par[xp] = yp
        
        else:
            self.par[xp] = yp
            self.rnk[yp]+=1
            
        # without rank
        # self.par[xp] = yp
        
 dsu = DSU(1000)
 dsu.find(x)
 dsu.union(x,y)

```

## 7. General
### Queue
```python
from collections import deque
q = dequeu()

q.popleft()
q.pop()
q.append()
```
### Custom Object Comparison & Heap compare
```python
import heapq

class Node:
    
    def __init__(self,val1,val2):
        self.val1 = val1
        self.val2 = val2
        
    def __lt__(self,other):
        
        if self.val1 < other.val1:
            return True
        elif self.val1 == other.val1:
            return self.val2 < other.val2
        
        return False
    
    def __str__(self):
        return str(self.val1)+" "+str(self.val2)

heap = []
heap.append(Node(20,3))
heap.append(Node(4,5))
heap.append(Node(4,3))
heap.append(Node(4,1))
heap.append(Node(2,3))
heap.append(Node(3,5))


heapq.heapify(heap)

while heap:
    print(heapq.heappop(heap))

```
### Custom Sorting
```python
>>> dicto
{'a': 21, 's': 12, 'b': 21, 'c': 21}
>>> dicto.items()
[('a', 21), ('s', 12), ('b', 21), ('c', 21)]
>>> sorted(dicto.items(),key=lambda x: (-x[1],x[0]))
[('a', 21), ('b', 21), ('c', 21), ('s', 12)]
```
* More Complex Sorting - https://docs.python.org/3/howto/sorting.html#sort-stability-and-complex-sorts
### Max Int, Min Int
```python
sys.maxsize, -sys.maxsize-1
float('inf'), -float('inf') 
```



# B. Algorithm Implementations

## 1. Binary-Search
### Left-Most or just smaller value
```python

# first x for which func(x) is true
def first_equal_greater(lo, hi, func, val):
    # set lo = 1 or 0 in most cases
    while lo < hi:
        mid = (lo +hi)//2
	
	if func(mid) == True: #if nums[mid] >= val:
	    hi = mid
	else:
	    lo = mid+1

    if func(lo) == False:
	return -1

    return lo

```

### Right Most or just greater value

```python
# last x for which func(x) is True
def last_greater(lo, hi, func, val):
	
    while lo < hi:
	mid = (lo + hi +1)//2
    
    if func(mid) == True: #if nums[mid] <= val:
    	lo = mid
    else:
	hi = mid-1

    if func(hi) == False:
	return -1
    
    return hi

```

## 2. Quick-Select

## 3. Sorting
### Quick-Sort

### Merge-Sort

## 4. Graphs
### DFS
```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        
        dd = [(0,1),(0,-1),(1,0),(-1,0)]
        vis = {}
        
        check_valid  = lambda x,y: x < len(grid) and x >=0 and y >=0 and y < len(grid[0]) 

        def dfs(i,j,graph):
            
            vis[(i,j)] = 1
            
            for dx,dy in dd:
                x = i+dx
                y = j+dy
                if (x,y) not in vis and check_valid(x,y) and graph[x][y] == '1':
                    dfs(x,y,graph)
                
            
        cnt=0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                
                if (i,j) not in vis and grid[i][j] == '1':
                    dfs(i,j,grid)
                    cnt+=1
        
        return cnt
```
### BFS

### Dijkstra

### Topological Sorting

### Bellman-Ford

### Floyd-Warshal

### Strongly-Connected Components
```python
 def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        
        
        low = [-1]*n
        disc = [-1]*n
        parent = [-1]*n
        res = []
        
        graph = defaultdict(list)
        
        for tup in connections:
            
            graph[tup[0]].append(tup[1])
            
            graph[tup[1]].append(tup[0])
        
        self.time = 1
        for i in range(n):
            if disc[i] == -1:
                self.dfs(i, graph, disc, low, res, parent)
        
        
        return res
                
                
    def dfs(self,u, graph, disc, low, res,parent):
        
        disc[u] = self.time
        low[u] = self.time
        
        self.time+=1
        #children=0
        for v in graph[u]:
            
            if disc[v] == -1:
                parent[v] = u
                #children+=1
                self.dfs(v, graph, disc, low, res,parent)
                
                low[u] = min(low[u], low[v])
                
                if low[v] > disc[u]:
                # u - v is critical, there is no path for v to reach back to u or previous vertices of u
                    res.append([u,v])
                
                # Articulation Points
                '''
                # u is an articulation point in following cases 
                # (1) u is root of DFS tree and has two or more chilren. 
                if parent[u] == -1 and children > 1: 
                    res.append(u)
  
                #(2) If u is not root and low value of one of its child is more 
                # than discovery value of u. 
                if parent[u] != -1 and low[v] >= disc[u]: 
                    res.append(u)    
                '''
            elif v!= parent[u]:
                
                low[u] = min(low[u], disc[v])
        
```

## 5. String Matching

### KMP






