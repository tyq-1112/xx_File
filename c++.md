# c++

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=1e5+5,M=2e4+5,inf=0x3f3f3f3f,mod=1e9+7;
const int dir[][2]={{0, 1}, {1, 0}, {0, -1}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
#define lowbit(x) ((x) & (-x))
#define mst(a,b) memset(a,b,sizeof a)
#define PII pair<int,int>
#define fi first
#define se second
#define pb push_back
unordered_map<int, int> ump;
map<int,int>mp;
#define rep(i, a, n) for (ll i = a; i <= n; i++)
#define per(i, n, a) for (ll i = n; i >= a; i--)
#define bug(x) cout << #x << "===" << x << endl
#define Please return
#define AC 0 
ll f[N];
int find(int x){return x==f[x]?x:f[x]=find(f[x]);}
void mer(int x,int y){f[find(x)] = f[find(y)];}
//string str_rev(str.rbegin(), str.rend());
ll gcd(ll a,ll b) {return b == 0 ? a : gcd(b, a % b);}
ll Pow(ll a, ll b){ll ans = 1;while(b > 0){if(b & 1){ans = ans * a % mod;}a = a * a % mod;b >>= 1;}return ans;}
inline int read(){int x = 0, f = 1;char ch = getchar();
    while(ch < '0' || ch > '9'){if (ch == '-')f = -1;ch = getchar();}
    while(ch >= '0' && ch <= '9'){x = (x<<1) + (x<<3) + (ch^48);ch = getchar();}return x * f;}

int main(){
    ios::sync_with_stdio(0);cin.tie(0); 
    
    Please AC;
}
```

# 算法基础：

### 二分模板

```c++
bool check(int x) {/* ... */} // 检查x是否满足某种性质
// 区间[l, r]被划分成[l, mid]和[mid + 1, r]时使用：
int bsearch_1(int l, int r)
{
    while (l < r)
    {
        int mid = l + r >> 1;
        if (check(mid)) r = mid;    // check()判断mid是否满足性质
        else l = mid + 1;
    }
    return l;
}
// 区间[l, r]被划分成[l, mid - 1]和[mid, r]时使用：
int bsearch_2(int l, int r)
{
    while (l < r)
    {
        int mid = l + r + 1 >> 1;
        if (check(mid)) l = mid;
        else r = mid - 1;
    }
    return l;
}


//浮点数二分算法模板
bool check(double x) {/* ... */} // 检查x是否满足某种性质

double bsearch_3(double l, double r)
{
    const double eps = 1e-6;   // eps 表示精度，取决于题目对精度的要求
    while (r - l > eps)
    {
        double mid = (l + r) / 2;
        if (check(mid)) r = mid;
        else l = mid;
    }
    return l;
}

lower_bound(起始地址，结束地址，要查找的数值) 返回的是数值 第一个 出现的位置。

upper_bound(起始地址，结束地址，要查找的数值) 返回的是 第一个大于待查找数值 出现的位置。

binary_search(起始地址，结束地址，要查找的数值)  返回的是是否存在这么一个数，是一个bool值。
```

### 优先队列：中位数

> ```c++
> 小： priority_queue<int, vector<int>, greater<int> > q; 
> 
> 大： priority_queue <int,vector<int>,less<int> >q;
> 
> vector去重：alls.erase(unique(alls.begin() , alls.end()), alls.end());
> 
> class MedianFinder {
> public:
> 
>     priority_queue<int> da;
>     priority_queue<int,vector<int>,greater<int>> xi;
>     
>     void addNum(int num) {
>         if(!da.size() || num < da.top()) da.push(num);
>         else xi.push(num);
> 
>         if(da.size() == xi.size()+2) {
>             xi.push(da.top());
>             da.pop();
>         }else if(xi.size() == da.size()+1){
>             da.push(xi.top());
>             xi.pop();
>         }
>     }
>     
>     double findMedian() {
>         int q = da.size() , qq = xi.size();
>         if((q+qq)&1) return da.top();
>         return (da.top() + xi.top()) /2.0;
>     }
> };
> ```

### [优先队列：看病要排队](http://acm.hdu.edu.cn/showproblem.php?pid=1873)

```c++
/*医生在看病时，则会在他的队伍里面选择一个优先权最高的人进行诊治。如果遇到两个优先权一样的病人的话，则选择最早来排队的病人。
一共有两种事件：
1:"IN A B",表示有一个拥有优先级B的病人要求医生A诊治。(0<A<=3,0<B<=10)
2:"OUT A",表示医生A进行了一次诊治，诊治完毕后，病人出院。(0<A<=3)
对于每个"OUT A"事件，请在一行里面输出被诊治人的编号ID。如果该事件时无病人需要诊治，则输出"EMPTY"。
诊治人的编号ID的定义为：在一组测试中，"IN A B"事件发生第K次时，进来的病人ID即为K。从1开始编号。*/

#include <iostream>
#include <queue>
 
using namespace std;
 
const int N = 3;
 
struct Node {
    int p;      // 优先级
    int id;     // 病人ID（序号）
    friend bool operator <(const Node &a,const Node &b) {
    	//如果优先级一样，先看编号最前的，否则看优先级高的 
        return a.p == b.p ? a.id > b.id : a.p < b.p;
    }
};
 
void solve(int n)
{
    priority_queue<Node> q[N];
    string op;
    Node t;
    int a, b, k;
 
    k = 0;
    for(int i=1; i<=n; i++) {
        cin >> op;
        if(op == "IN") {
            cin >> a >> b;
            t = {b,++k};
            q[a - 1].push(t);
        } else if(op == "OUT") {
            cin >> a;
            if(q[a - 1].empty())
                cout << "EMPTY" << endl;
            else {
                cout << q[a - 1].top().id << endl;
                q[a - 1].pop();
            }
        }
    }
}
 
int main()
{
    int n;
    while(cin >> n) {
        solve(n);
    }
}
```

### [归并排序](https://ac.nowcoder.com/acm/problem/20861)

```c++
/*
一个逆序对(i,j) 需要满足 i < j 且 ai > aj
兔子可以把区间[L,R] 反转，例如序列{1,2,3,4} 反转区间[1,3] 后是{3,2,1,4}。
兔子有m次反转操作，现在兔子想知道每次反转后逆序对个数是奇数还是偶数，兔子喜欢偶数，而讨厌奇数。
请注意，每一次反转操作都会对原序列进行改变。例如序列{1,2,3,4} 第一次操作区间[1,2] 后变成{2,1,3,4} 第二次反转区间[3,4] 后变成 {2,1,4,3}
*/
#include<bits/stdc++.h>
using namespace std;
const int maxn = 1e5+5;
int a[maxn]  , b[maxn], n , m , l , r ;
long long int ans;

void _merge(int l , int mid , int r){
    int p1 = l , p2 = mid+1;
    for(int i=l;i<=r;i++){
		if(p1<=mid&&((p2>r)||(a[p1]<=a[p2]))){
			b[i]=a[p1];
			p1++;
		}
		else {
			b[i]=a[p2];
			p2++; 
			ans+=mid-p1+1;
		}
	}
    for(int i = l ; i<=r;i++) a[i] = b[i];
}
void check(int l , int r){
    int mid = l+(r-l)/2;
    if(l<r){
        check(l,mid);
        check(mid+1 , r);
    }
    _merge(l,mid,r);
}
int main(){
    scanf("%d",&n);
    for(int i=1;i<=n;i++) scanf("%d",&a[i]);
    check(1,n);
    ans = ans%2;
    m = read();
    while(m--){
        scanf("%d %d",&l,&r);
        int p = r-l+1;
        if ((p*(p-1)/2)&1) //判断区间反转次数p*(p-1)/2的奇偶 
            ans ^= 1;//改变奇偶性 
        
        if (ans) puts("dislike");
        else puts("like");
    }
}
```

### [快速排序](https://ac.nowcoder.com/acm/problem/207028)

```c++
//求第k小的数
#include <bits/stdc++.h>
using namespace std;

const int N=5e6+6;
int a[N] , t, n ,k;

int qksort(int l,int r,int k)
{
	if(l==r) return a[l];
    int i=l,j=r;
    int mid=l+(r-l)/2;
	int x=a[mid];
    while(i<=j)
    {
        while(a[j]>x) j--;
        while(a[i]<x) i++;
    	if(i<=j)
    	{
    		swap(a[i],a[j]);
    		i++,j--;
		}
    }
    if(k<=j) return qksort(l,j,k);
    else if(k>=i) return qksort(i,r,k);
    else return a[k];
}

int main()
{
    scanf("%d",&t);
    while(t--)
    {
        scanf("%d %d",&n,&k);
        for(int i=1;i<=n;i++) scanf("%d",&a[i]);
        printf("%d\n",qksort(1,n,k));
    }
    
    return 0;
}
```

### 基数排序

```c++
class Solution {
public:
    int maximumGap(vector<int>& nums) {
        int n = nums.size();
        if (n < 2) {
            return 0;
        }
        int exp = 1;
        vector<int> buf(n);
        int maxVal = *max_element(nums.begin(), nums.end());

        while (maxVal >= exp) {
            vector<int> cnt(10);
            for (int i = 0; i < n; i++) {
                int digit = (nums[i] / exp) % 10;
                cnt[digit]++;
            }
           
            for (int i = 1; i < 10; i++) {
                cnt[i] += cnt[i - 1];
            } 

            for (int i = n - 1; i >= 0; i--) {
                int digit = (nums[i] / exp) % 10;
                buf[cnt[digit] - 1] = nums[i];
                cnt[digit]--;
            }
            copy(buf.begin(), buf.end(), nums.begin());
            exp *= 10;
        }
    }
};
```

### 前缀和与差分

```c++
//一维前缀和 
S[i] = a[1] + a[2] + ... a[i]
a[l] + ... + a[r] = S[r] - S[l - 1]

//二维前缀和
S[i, j] = 第i行j列格子左上部分所有元素的和
初始化：S[i][j] = S[i-1][j] + S[i][j-1] - S[i-1][j-1] + a[i][j]
以(x1, y1)为左上角，(x2, y2)为右下角的子矩阵的和为：
S[x2, y2] - S[x1 - 1, y2] - S[x2, y1 - 1] + S[x1 - 1, y1 - 1]

    
//一维差分 
给区间[l, r]中的每个数加上c：B[l] += c, B[r + 1] -= c

//二维差分
给以(x1, y1)为左上角，(x2, y2)为右下角的子矩阵中的所有元素加上c：
S[x1, y1] += c, S[x2 + 1, y1] -= c, S[x1, y2 + 1] -= c, S[x2 + 1, y2 + 1] += c
```

### 离散：前缀和

```c++
/*假定有一个无限长的数轴，数轴上每个坐标上的数都是 0。
现在，我们首先进行 n 次操作，每次操作将某一位置 x 上的数加 c。
接下来，进行 m 次询问，每个询问包含两个整数 l 和 r，你需要求出在区间 [l,r] 之间的所有数的和。
−10^9≤x≤10^9 ,
1≤n,m≤10^5,
−10^9≤l≤r≤10^9,
−10000≤c≤10000
*/

#include<bits/stdc++.h>
using namespace std;
const int N = 1e6+6;
typedef pair<int,int> PII;
int n,m,c,cc;
int s[N],a[N];
vector<int> vec;      //所有用到的点
vector<PII> vm,vn;   //离线操作

int find(int x){
    int l=0 , r = vec.size();
    while(l<r){
        int mid  = l+r>>1;
        if(vec[mid] < x) l = mid+1;
        else r = mid;
    }
    return r+1;
}
int main(){
    cin>>n>>m;
    while(n--){
        cin>>c>>cc;
        vec.push_back(c);
        vn.push_back({c,cc});
    }
    while(m--){
        cin>>c>>cc;
        vm.push_back({c,cc});
        vec.push_back(c);
        vec.push_back(cc);
    }
    
    sort(vec.begin() , vec.end());
    vec.erase(unique(vec.begin(),vec.end()) , vec.end());
    
    for(auto temp : vn){
        int x = temp.first;
        a[find(x)]+=temp.second;
    }
    
    for(int i=1;i<=vec.size();i++) s[i] = s[i-1]+a[i];
    
    for(auto temp : vm){
        int l = temp.first;
        int r = temp.second;
        l = find(l);
        r = find(r);
        cout<<s[r]-s[l-1]<<endl;
    }
}
```



### 二维差分

```c++
#include <iostream>
#include <cstdio>
using namespace std;
const int maxn = 1e3 + 40;
int a[maxn][maxn], b[maxn][maxn];

inline void insert(int x1, int y1, int x2, int y2, int c) {
    b[x1][y1] += c;
    b[x2 + 1][y1] -= c;
    b[x1][y2 + 1] -= c;
    b[x2 + 1][y2 + 1] += c;
}

int main() {
    int n, m, q;
    scanf("%d%d%d", &n, &m, &q);
    for(int i = 1; i <= n; i++)
        for(int j = 1; j <= m; j++)
            scanf("%d", &a[i][j]);

    for(int i = 1; i <= n; i++)
        for(int j = 1; j <= m; j++)
            insert(i, j, i, j, a[i][j]);

    for(int i = 1; i <= q; i++) {
        int x1, y1, x2, y2, c;
        scanf("%d%d%d%d%d", &x1, &y1, &x2, &y2, &c);
        insert(x1, y1, x2, y2, c);
    }

    for(int i = 1; i <= n; i++)
        for(int j = 1; j <= m; j++)
            b[i][j] += b[i - 1][j] + b[i][j - 1] - b[i - 1][j - 1];

    for(int i = 1; i <= n; i++)
        for(int j = 1; j <= m; j++)
            if(j == m) printf("%d\n", b[i][j]);
            else printf("%d ", b[i][j]);
            
    return 0;
}

```



### 位运算

```c++
右移一位：x>>1
    
在最后加一个0：x<<1
    
最最后加一个1：(x<<1)+1
    
把最后一位变成1：x|1
    
把最后一位变成0：(x|1)-1
    
最后一位取反：x^1

把右数第k位变成1：x|(1<<(k-1))
    
把右数第k位变成0：x&(~(1<<(k-1)))
    
右数第k位取反：x^(1<<(k-1))
    
求n的第k位数字: n >> k & 1
    
返回n的最后一位1：lowbit(n) = n & -n
   
二进制1的个数： __builtin_popcount(n);
```

# 异或

### [最大异或对](https://www.acwing.com/problem/content/145/)

```c++
//在给定的 N 个整数 A1，A2……AN 中选出两个进行 xor（异或）运算，得到的结果最大是多少？
//1≤N≤10^5,0≤Ai<2^31

#include<bits/stdc++.h>
using namespace std;

const int N = 1e5+5 , M = N*30;
int son[M][2] , a[N] , n , idx;

void insert(int x){
    int p = 0;
    for(int i=30;~i;i--){
        int &s = son[p][x>>i&1];
        if(!s) s = ++idx;
        p = s;
    }
}

int query(int x){
    int p=0 , res = 0;
    for(int i=30;~i;i--){
        int s = x>>i&1;
        if(son[p][!s]){
            p = son[p][!s];
            res+=1<<i;
        }else p = son[p][s];
    }
    return res;
}

int main()
{
    cin>>n;
    for(int i=0;i<n;i++) cin>>a[i] , insert(a[i]);
    int res = 0;
    for(int i=0;i<n;i++) res = max(res , query(a[i]));
    cout<<res<<endl;
    return 0;
}

```



# 高精度

### [加，减，乘，除](https://www.luogu.com.cn/problem/P2142)

```c++
#include<bits/stdc++.h>
using namespace std;
string a,b;
vector<int> A , B ,C;

void add(vector<int> &A, vector<int> &B){
	if (A.size() < B.size()) return add(B, A);
	int t = 0;
	for(int i=0;i<A.size();i++){
		t+=A[i];
		if(i<B.size()) t+=B[i];
		C.push_back(t%10);
		t/=10;
	}
	if(t) C.push_back(t);
}

bool cmp(string a , string b){
	if(a.size() > b.size()) return true;
	for(int i=0;i<a.size();i++) if(a[i]<b[i]) return false;
	return true;
} 

void sub(vector<int> &A, vector<int> &B){
	int t = 0;
	for(int i=0;i<A.size();i++){
		t = A[i]-t;
		if(i<B.size()) t-=B[i];
		C.push_back((t+10)%10);
		if(t<0) t=1;
		else t = 0;
	}
	while(C.size()>1 && C.back() == 0) C.pop_back();
}
// 高精度乘低精度
void mul(vector<int> &A, int b)
{
    int t = 0;
    for (int i = 0; i < A.size() || t; i ++ )
    {
        if (i < A.size()) t += A[i] * b;
        C.push_back(t % 10);
        t /= 10;
    }void
}
// A / b = C ... r, A >= 0, b > 0
void div(vector<int> &A, int b, int &r)
{
    r = 0;
    for (int i = A.size() - 1; i >= 0; i -- )
    {
        r = r * 10 + A[i];
        C.push_back(r / b);
        r %= b;
    }
    reverse(C.begin(), C.end());
    while (C.size() > 1 && C.back() == 0) C.pop_back();
}

int main(){
	cin>>a>>b;
	for(int i=a.size()-1;i>=0;i--) A.push_back(a[i]-'0');
	for(int i=b.size()-1;i>=0;i--) B.push_back(b[i]-'0');
	
//	add(A,B);
	
	if(cmp(a,b)) sub(A,B);
	else sub(B,A) , cout<<"-";
	
	for(int i=C.size()-1;i>=0;i--) cout<<C[i];
	return 0;
} 
```

### 大数加乘

```c++
string add(string a,string b)//只限两个非负整数相加
{
    string ans;
    int na[L]={0},nb[L]={0};
    int la=a.size(),lb=b.size();
    for(int i=0;i<la;i++) na[la-1-i]=a[i]-'0';
    for(int i=0;i<lb;i++) nb[lb-1-i]=b[i]-'0';
    int lmax=la>lb?la:lb;
    for(int i=0;i<lmax;i++) na[i]+=nb[i],na[i+1]+=na[i]/10,na[i]%=10;
    if(na[lmax]) lmax++;
    for(int i=lmax-1;i>=0;i--) ans+=na[i]+'0';
    return ans;
}
 
string mul(string a,string b)//高精度乘法a,b,均为非负整数
{
    string s;
    int na[L],nb[L],nc[L],La=a.size(),Lb=b.size();//na存储被乘数，nb存储乘数，nc存储积
    fill(na,na+L,0);fill(nb,nb+L,0);fill(nc,nc+L,0);//将na,nb,nc都置为0
    for(int i=La-1;i>=0;i--) na[La-i]=a[i]-'0';//将字符串表示的大整形数转成i整形数组表示的大整形数
    for(int i=Lb-1;i>=0;i--) nb[Lb-i]=b[i]-'0';
    for(int i=1;i<=La;i++)
        for(int j=1;j<=Lb;j++)
        nc[i+j-1]+=na[i]*nb[j];//a的第i位乘以b的第j位为积的第i+j-1位（先不考虑进位）
    for(int i=1;i<=La+Lb;i++)
        nc[i+1]+=nc[i]/10,nc[i]%=10;//统一处理进位
    if(nc[La+Lb]) s+=nc[La+Lb]+'0';//判断第i+j位上的数字是不是0
    for(int i=La+Lb-1;i>=1;i--)
        s+=nc[i]+'0';//将整形数组转成字符串
    return s;
}
```

# 数据结构：

### [表达式计算](https://ac.nowcoder.com/acm/problem/50999)

```c++
#include<bits/stdc++.h>
using namespace std;
string str;
int getnum(int l , int r){
    int num=0;
    for(int i=l;i<=r;i++){
        num = num*10+str[i]-'0';
    }
    return num;
}
int calc(int l , int r){
    if(l>r) return 0;
    int pos1 = -1 , pos2 = -1 , pos3 = -1 ;   //优先级
    int cnt = 0 ;    //记括号
    for(int i = l ; i <=r ; i++){
        if(str[i]=='(') cnt++;
        if(str[i]==')') cnt--;
        if(cnt==0 && (str[i]=='+'||str[i]=='-')) pos1 = i;
        if(cnt==0 && (str[i]=='*'||str[i]=='/')) pos2 = i;
        if(cnt==0 && (str[i]=='^')) pos3 = i;
    }
    
    if(pos1 == -1 && pos2 ==-1 && pos3 ==-1){    //括号不匹配，有多
        if(cnt>0&&str[l]=='(') return calc(l+1 , r);
        else if(cnt<0 && str[r]==')') return calc(l,r-1);
        else if(cnt==0 && str[r]==')' && str[l]=='(') return calc(l+1 , r-1);
        else return getnum(l,r);
    }
    
    if(pos1!=-1){
        if(str[pos1]=='+') return calc(l,pos1-1) + calc(pos1+1,r);
        if(str[pos1]=='-') return calc(l,pos1-1) - calc(pos1+1,r);
    }
    else if(pos2!=-1){
        if(str[pos2]=='*') return calc(l,pos2-1) * calc(pos2+1,r);
        if(str[pos2]=='/') return calc(l,pos2-1) / calc(pos2+1,r);
    }
    else if(pos3!=-1){
        return (int)pow(calc(l,pos3-1),calc(pos3+1,r));
    }
    return 0;
}
int main(){
    cin>>str;
    cout<< calc(0,str.length()-1) <<endl;
    return 0;
}
```

### [ 相对分子质量](https://ac.nowcoder.com/acm/contest/11163/B)

```c++
/*
第一行中给出两个正整数M,N,,1≤M≤100,1≤N≤20
接下来M行每行给出一个以大写字母开头的字符串S和一个正整数X，S为元素名称，X为相对原子质量,1≤X≤500
最后N行每行给出一个化学式，保证化学式长度不超过50个字符
元素可能是一个大写字母，也可能是一个大写字母跟着一个小写字母，保证给出的M种化学元素互不相同
化学式包含括号以及括号嵌套，例如：Ba((OH)2(CO3)2)3
2 2
H 1
O 16
H2
H2O
res：
2
18
*/
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
int n,m;
map<string , ll > mp;
string str;

void slove(){
	ll ans = 0;
	int rear = 0;  
	ll st[5050] = {0};   
	str = str+"#";
	for(int i=0;i<str.length();i++){
		if(str[i]==')'){
			//(Aa2Ab2)10 
			ll sum=0;
			while(st[rear]!=-1)sum+=st[rear--];
			rear--;
			ll num = 0;
			while(str[i+1]>='0'&&str[i+1]<='9') i++,num=num*10+str[i]-'0';
			if(!num) num=1;
			st[++rear] = sum*num;
			
		}else if(str[i]=='('){
			st[++rear] = -1;
			
		}else if(str[i]>='A'&&str[i]<='Z'){
			string s = "";
			s = s+str[i];
			while(str[i+1]>='a'&&str[i+1]<='z') i++,s+=str[i];
			ll num = 0;
			while(str[i+1]>='0'&&str[i+1]<='9') i++,num=num*10+str[i]-'0';
			if(!num) num = 1;
			//cout<<mp[s]<<" "<<num<<endl;
			st[++rear] = num*mp[s];
		}
		
	}
	while(rear) ans+=st[rear--];
	cout<<ans<<endl;
}
int main(){
	cin>>m>>n;
	string s; ll x;
	while(m--){
		cin>>s>>x;
		mp[s]  = x;
	}
	while(n--){
		cin>>str;
		slove();
	}
	return 0;
}
 
```



### [单调栈：小A的柱状图](https://ac.nowcoder.com/acm/problem/23619)

```c++
#include<bits/stdc++.h>
typedef long long ll;
const int maxn =1e6+5;
using namespace std;
int n,h[maxn],wid[maxn],le[maxn],ri[maxn],a;
//h代表每个矩形的高度，a代表矩形的宽度，wid是从1到第i个矩形的总宽度，
// le ri能到达最左或者最右的第几个矩形 

stack<int>q;
long long ans =0;
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);cout.tie(0);
    cin>>n;
    for(int i=1;i<=n;i++)
    {
        cin>>a;
        wid[i]=wid[i-1]+a;
    }
    
    for(int i=1;i<=n;i++) cin>>h[i];
    
    for(int i=1;i<=n;i++)//找每个矩形能到达最左 
    {
        while(!q.empty()&&h[q.top()]>=h[i]) q.pop();
        if(q.empty()) le[i]=1;
        else le[i]=q.top()+1;
        q.push(i);
    }
    while(!q.empty()) q.pop();
    
    for(int i=n;i>=1;i--) //找每个矩形能到达最右 
    {
        while(!q.empty()&&h[q.top()]>=h[i]) q.pop();
        if(q.empty()) ri[i]=n;
        else ri[i]=q.top()-1;
        q.push(i);
    }
    
    for(int i=1;i<=n;i++) //找最大值 
    {
        ans =max (ans,(1LL)*(wid[ri[i]]-wid[le[i]-1])*h[i]);
    }
    cout<<ans<<endl;
    return 0;
}
```

### [滑动窗口：](https://ac.nowcoder.com/acm/problem/50528)

```c++
/*n个数，求每k个数的最大值最小是
8 3
1 3 -1 -3 5  3 6 7

最小值：-1 -3 -3 -3 3 3
最大值： 3 3 5 5 6 7
*/
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N = 1e6+6;
int q[N] , l , r= -1 , n , k , a[N];

int main(){
    cin>>n>>k;
    for(int i=1;i<=n;i++) cin>>a[i];
    
    for(int i=1;i<=n;i++){
        while( r>=l && a[q[r]] >= a[i]) r--;
        q[++r] = i;
        if(i-k+1>q[l]) l++;
        if(i>=k) cout<<a[q[l]]<<" ";
    }
    
    puts("");
    l = 0 ; r = -1;
    for(int i=1;i<=n;i++){
        while( r>=l && a[q[r]] <= a[i]) r--;
        q[++r] = i;
        if(i-k+1>q[l]) l++;
        if(i>=k) cout<<a[q[l]]<<" ";
    }
    return 0;
}
```

### [并查集：食物链](https://ac.nowcoder.com/acm/problem/16884)

```c++
/*
动物王国中有三类动物A,B,C，这三类动物的食物链构成了有趣的环形。A吃B，B吃C，C吃A。
现有N个动物，以1－N编号。每个动物都是A,B,C中的一种，但是我们并不知道它到底是哪一种。
有人用两种说法对这N个动物所构成的食物链关系进行描述：
第一种说法是“1 X Y”，表示X和Y是同类。
第二种说法是“2 X Y”，表示X吃Y。
此人对N个动物，用上述两种说法，一句接一句地说出K句话，这K句话有的是真的，有的是假的。当一句话满足下列三条之一时，这句话就是假话，否则就是真话。
1） 当前的话与前面的某些真的话冲突，就是假话；
2） 当前的话中X或Y比N大，就是假话；
3） 当前的话表示X吃X，就是假话。
你的任务是根据给定的N（1≤N≤50,000）和K句话（0≤K≤100,000），输出假话的总数。
*/

//开三倍的数组，1-n表示与x同类，n+1到2n表示x吃的动物，2n+1到3*n表示吃x的动物。
//后每次直接和并和查询即可。

#include<bits/stdc++.h>
using namespace std;
const int maxn = 3*5e4+10;
int fa[maxn] , n ,k;
int d,x , y , ans;
int find(int x)
{
    return x==fa[x]?x:find(fa[x]);
}
void mer(int x,int y)
{
    fa[find(x)] = fa[find(y)];
}
int main(){
    scanf("%d%d",&n,&k);
    for(int i=1;i<3*n+10;i++) fa[i] = i;
    while(k--){
        scanf("%d%d%d",&d,&x,&y);
        if(x>n || y>n) {ans++;continue;}   //当前的话中X或Y比N大，就是假话；
        
        if(d==1){     //表示X和Y是同类。
            if (find(x) == find(y+n) || find(x) == find(y+2*n))
            {
                ans++;
                continue;
            }
            mer(x, y);
            mer(x+n, y+n);
            mer(x+2*n, y+2*n);
        }else{        //表示X吃Y。
            if (find(x) == find(y) || find(x) == find(y+2*n))
            {
                ans++;
                continue;
            }
            mer(x, y+n);
            mer(x+n, y+2*n);
            mer(x+2*n, y);
        }
    }
    cout<<ans;
}

//是用一个d数组记录当前动物与根节点的距离，只要距离相减余3等于0则为同类，余1则是x吃y。只要两个动物有关系就放在同一个集合中。

#include<bits/stdc++.h>
using namespace std;
const int N=5e4+10;
int p[N],d[N];
 
int find(int x){
    if(p[x]!=x){
        int t=find(p[x]);
        d[x]+=d[p[x]];
        p[x]=t;
    }
    return p[x];
}
 
int main(){
    int n,m;
    cin>>n>>m;
    for(int i=1;i<=n;i++)
        p[i]=i;
    int res=0;
    while(m--){
        int t,x,y;
        cin>>t>>x>>y;
        if(x>n||y>n) res++;
        else{
            int px=find(x),py=find(y);
            if(t==1){
                if(px==py&&(d[x]-d[y])%3) res++;
                else if(px!=py){
                    p[px]=py;
                    d[px]=d[y]-d[x];
                }
            }
            else{
                if(px==py&&(d[x]-d[y]-1)%3) res++;
                else if(px!=py){
                    p[px]=py;
                    d[px]=d[y]-d[x]+1;
                }
            }
        }
    }
    cout<<res<<endl;
    return 0;
}

```



### [二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/solution/)

```c++
#include<bits/stdc++.h>
using namespace std;
vector<int> pre , in , post , ans , ceng;
map<int,int> mp;
struct TreeNode{
	int val;
	TreeNode *left , *right;
	TreeNode(int x):val(x),left(NULL),right(NULL){}
};
//先序和中序得到二叉树
TreeNode* build(vector<int>& pre,vector<int> & in,int preL,int preR,int inL,int inR){
	if(preL > preR || inL > inR) return NULL;
	TreeNode* root = new TreeNode(pre[preL]);  //先序的第一个节点就是根 
	int index = mp[pre[preL]];  //找到根在中序的位置 
	int left_size = index - inL; //得到左子树中的节点数目
	root->left = build(pre ,in, preL+1 , preL+left_size , inL , index-1);
	root->right = build(pre ,in, preL+left_size+1 , preR , index+1 , inR);
	return root;
}

TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    int n = preorder.size();
    for(int i=0;i<n;i++) mp[inorder[i]] = i;
    return build(preorder,inorder,0,n-1,0,n-1); 
}
/*
中序和后序得到二叉树
TreeNode* build(vector<int>& in , vector<int> & post , int inL , int inR  , int postL , int postR){
	if(inL > inR || postL > postR) return NULL;
	TreeNode* root = new TreeNode(post[postR]);  //后序的最后一个节点就是根 
	int index = mp[post[postR]];  //找到根在中序的位置 
	int left_size = index - inL; //得到左子树中的节点数目
	root->left = build(in ,post, inL , index-1 , postL , postL+left_size-1);
	root->right = build(in ,post, index+1 , inR , postL+left_size , postR-1);
	return root;
}
TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
    int n = inorder.size();
    for(int i=0;i<n;i++) mp[inorder[i]] = i;
    return build(inorder,postorder,0,n-1,0,n-1); 
}
*/
/*
9
in:  4 7 2 1 8 5 9 3 6
post:7 4 2 8 9 5 6 3 1
ans: 1 2 4 7 3 5 8 9 6
*/
//后序遍历 
void postorder(TreeNode *root){
	if(root){
		postorder(root->left);
		postorder(root->right);
		ans.push_back(root->val);
	}
}
//层序遍历
void bfsorder(TreeNode &root){
	queue<TreeNode> q;
	q.push(root);
	while(!q.empty()){
		TreeNode now = q.front();q.pop();
		ceng.push_back(now.val);
		if(now.left!=NULL) q.push(*now.left);
		if(now.right!=NULL) q.push(*now.right);
	}
	for(int i=0;i<ceng.size();i++) printf("%d%c",ceng[i],i==ceng.size()-1?'\n':' ');
} 

int main(){
	int n , data; cin>>n;
	for(int i=0;i<n;i++) {
		scanf("%d",&data); pre.push_back(data);
	}
	for(int i=0;i<n;i++) {
		scanf("%d",&data); in.push_back(data);
	}
	TreeNode *root = buildTree(pre , in);
	postorder(root);
	for(int i=0;i<ans.size();i++) printf("%d%c",ans[i],i==ans.size()-1?'\n':' ');
	//bfsorder(*root);
} 
```

### rope：

```c++
#include<bits/stdc++.h>
#include<ext/rope>
  
using namespace std;
using namespace __gnu_cxx;
  
rope<int> card;
rope<int> test ; 
int main(){
    int n,m;
    cin>>n>>m;
    for(int i=1;i<=n;++i){
        card.push_back(i);
    }
    while(m--){
        int f,s;
        cin>>f>>s;
        card=card.substr(f-1,s)+card.substr(0,f-1)+card.substr(f+s-1,n-f-s+1);
    }
    
    for(int i=0;i<n;++i) cout<<card[i]<<" ";
      
}

 Rope其主要是结合了链表和数组各自的优点，链表中的节点指向每个数据
 块，即数组，并且记录数据的个数，然后分块查找和插入。在g++头文件中，< ext / rope >中有成型的块状链表，在using namespace 
 __gnu_cxx;空间中，其操作十分方便。 
 rope test;
 test.push_back(x);//在末尾添加x
 test.insert(pos,x);//在pos插入x　　
 test.erase(pos,x);//从pos开始删除x个
 test.copy(pos,len,x);//从pos开始到pos+len为止用x代替
 test.replace(pos,x);//从pos开始换成x
 test.substr(pos,x);//提取pos开始x个
 test.at(x)/[x];//访问第x个元素

```



# 动态规划：

### [线性：编辑距离](https://leetcode-cn.com/problems/edit-distance/)

```c++
#include<bits/stdc++.h>
using namespace std;
string str1,str2;

int minDistance(string word1,string word2){
	int len1 = word1.length(), len2 = word2.length();
	if(len1==0 || len2==0) return max(len1 , len2);
	int dp[len1+1][len2+1];
	
	for(int i=0;i<=len1;++i) dp[i][0] = i;
	for(int i=0;i<=len2;++i) dp[0][i] = i;
	
	for(int i=1;i<=len1;++i){
		for(int j=1;j<=len2;j++){
			if(word1[i-1]==word2[j-1]) dp[i][j] = dp[i-1][j-1];
			else
				dp[i][j] = min(min(dp[i-1][j],dp[i][j-1]),dp[i-1][j-1])+1;
		}
	}
	return dp[len1][len2];
}

int main(){
	cin>>str1>>str2;
	cout<<minDistance(str1,str2)<<endl;
    //给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。
    //你可以对一个单词进行如下三种操作：插入一个字符删除一个字符替换一个字
} 
```

### [统计全1矩阵](https://leetcode-cn.com/problems/count-square-submatrices-with-all-ones/)

```c++

int countSquares(vector<vector<int>>& matrix) {
    int m = matrix.size(), n = matrix[0].size();
    vector<vector<int>> f(m, vector<int>(n));
    int ans = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == 0 || j == 0) {
                f[i][j] = matrix[i][j];
            }
            else if (matrix[i][j] == 0) {
                f[i][j] = 0;
            }
            else {
                f[i][j] = min(min(f[i][j - 1], f[i - 1][j]), f[i - 1][j - 1]) + 1;
            }
            ans += f[i][j];
        }
    }
    return ans;
}

```

### [统计全 1 子矩形](https://leetcode-cn.com/problems/count-submatrices-with-all-ones/)

```java
public int numSubmat(int[][] mat) {
        int n = mat.length;
        int m = mat[0].length;
        int[][] row = new int[n][m];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (j == 0) {
                    row[i][j] = mat[i][j];
                } else if (mat[i][j] != 0) {
                    row[i][j] = row[i][j - 1] + 1;
                } else {
                    row[i][j] = 0;
                }
            }
        }
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                int col = row[i][j];
                for (int k = i; k >= 0 && col != 0; --k) {
                    col = Math.min(col, row[k][j]);
                    ans += col;
                }
            }
        }
        return ans;
    }
/*输入：mat =[[1,0,1],
            [1,1,0],
            [1,1,0]]
输出：13
解释：
有 6 个 1x1 的矩形。
有 2 个 1x2 的矩形。
有 3 个 2x1 的矩形。
有 1 个 2x2 的矩形。
有 1 个 3x1 的矩形。
矩形数目总共 = 6 + 2 + 3 + 1 + 1 = 13 。*/
```



### [不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/)

```java
public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 1;
        // f(i) : 以i为根结点的bst个数
        // dp[n] : n个结点组成bst的个数
        // DP[n] = f(0) + f(1) + f(2) + ...f(n)
        // 当 i 为根节点时，其左子树节点个数为 i-1 个，右子树节点为 n-i，则 dp[n] = dp[i-1] * dp[n-i]
        for (int i = 2; i <= n; ++i) {
            for (int j = 1; j <= i; ++j) {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        return dp[n];
    }

输入: 3 输出: 5
解释:给定 n = 3, 一共有 5 种不同结构的二叉搜索树:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3

```

### PRE:毒瘤xor

```c++
/*
小a有N个数a1, a2, ..., aN，给出q个询问，每次询问给出区间[L, R]，现在请你找到一个数X，使得
亦或和最大
0<= x<=2^31
*/
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int maxn = 1e5+5;
int sum[maxn][32];    //维护前i个数字每一位有多少1
int main(){
    int n , data , q , l , r;
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>data;
        for(int j=0;j<31;j++){
            sum[i][j] = sum[i-1][j]+((data)>>j&1);
        }
    }
    cin>>q;
    while(q--){
        ll ans = 0;
        cin>>l>>r;
        for(int j=0;j<31;j++){
            int cnt = sum[r][j] - sum[l-1][j];   //区间数第j位1的个数
            int len = r-l+1;    //总个数
            //使区间和亦或最大，那么如果0多，我们更希望变为1，如果1多我们更希望不变
            // 0^1 = 1  1^1 = 0
            if(cnt < len-cnt) ans = ans+(1<<j);
        }
        cout<<ans<<endl;
    }
}
```

### [计数：整数划分](https://www.acwing.com/problem/content/902/)

```c++
/*一个正整数 n 可以表示成若干个正整数之和，形如：n=n1+n2+…+nk，其中 n1≥n2≥…≥nk,k≥1。
我们将这样的一种表示称为正整数 n 的一种划分。
现在给定一个正整数 n，请你求出 n 共有多少种不同的划分方法。
1≤n≤1000
*/

#include<bits/stdc++.h>
using namespace std;
const int N = 1e3+3 , mod = 1e9+7;
int dp[N] ,n;

int main()
{
    cin>>n;
    dp[0] = 1;
    // dp[i][j] : 表示前i个数拼成j的ans --->  完全背包
    for(int i=1;i<=n;i++){  
        for(int j=i;j<=n;j++){
            dp[j] = (dp[j]+dp[j-i])%mod;
        }
    }
    cout<<dp[n]<<endl;
    return 0;
}

#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1010, mod = 1e9 + 7;

int n;
int f[N][N];

int main()
{
    cin >> n;
   /*状态表示：f[i][j]表示总和为i，总个数为j的方案数
                            最小值为1的时候     大于1的时候
    状态转移方程：f[i][j] = f[i - 1][j - 1] + f[i - j][j];*/
    f[0][0] = 1;
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= i; j ++ )
            f[i][j] = (f[i - 1][j - 1] + f[i - j][j]) % mod;

    int res = 0;
    for (int i = 1; i <= n; i ++ ) res = (res + f[n][i]) % mod;

    cout << res << endl;

    return 0;
}
```

### [区间：环形石子合并](https://www.acwing.com/problem/content/284/)

```c++
/*
将 n 堆石子绕圆形操场排放，现要将石子有序地合并成一堆。
规定每次只能选相邻的两堆合并成新的一堆，并将新的一堆的石子数记做该次合并的得分。
4
4 5 9 4
res:
43
54
*/

#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N = 440;
int a[N] , n , s[N] , INF = 0x3f3f3f3f;
int f[N][N] , g[N][N];

int main()
{
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>a[i];
        a[i+n] = a[i];
    }
    for(int i=1;i<=2*n;i++) s[i] = a[i]+s[i-1];
    
    memset(f,0x3f,sizeof f);
    memset(g,-0x3f,sizeof g);
    
    for(int len = 1 ; len<=n; len++){
        for(int i=1;i+len-1<=2*n;i++){
            int l = i , r = i+len-1;
            if(len==1) f[l][r] = g[l][r] = 0;
            else{
                for(int k=l;k<r;k++){
                    f[l][r] = min(f[l][r] , f[l][k]+f[k+1][r]+s[r]-s[l-1]);
                    g[l][r] = max(g[l][r] , g[l][k]+g[k+1][r]+s[r]-s[l-1]);
                }
            }
        }
    }
    int ans = INF , res = -INF;
    for(int i=1;i<=n;i++){
        ans = min(ans , f[i][i+n-1]);//选择一种合并石子的方案，使得做 n−1 次合并得分总和最小。
        res = max(res , g[i][i+n-1]);//选择一种合并石子的方案，使得做 n−1 次合并得分总和最大。
    }
    cout<<ans<<endl<<res<<endl;
    return 0;
}
```

### [状压：最短Hamilton路径](https://www.acwing.com/activity/content/problem/content/1011/1/)

```c++
/*
给定一张 n 个点的带权无向图，点从 0∼n−1 标号，求起点 0 到终点 n−1 的最短 Hamilton 路径。
Hamilton 路径的定义是从 0 到 n−1 不重不漏地经过每个点恰好一次。
接下来 n 行每行 n 个整数，其中第 i 行第 j 个整数表示点 i 到 j 的距离（记为 a[i,j]）。
1≤n≤20
0≤a[i,j]≤107
*/

#include<bits/stdc++.h>
using namespace std;
const int N = 20 , M = 1<<N;
int dp[M][N];   //表示 i 号状态 ，必须经过 j 点
int mp[N][N];
int n;

int main(){
    cin>>n;
    for(int i = 0 ; i < n ; i++)
        for(int j = 0 ; j < n ; j++) cin>>mp[i][j];
    
    memset(dp, 0x3f, sizeof dp);
    dp[1][0] = 0;
    for(int i=0;i<(1<<n);i++){   //所有的状态
        for(int j=0;j<n;j++){
            if(i&(1<<j)){      //经过j号点
                for(int k=0 ; k<n ; k++){
                    if(i>>k&1) dp[i][j] = min(dp[i][j] , dp[i^(1<<j)][k] + mp[k][j]);
                }
            }
        }
    }
    
    cout<<dp[(1<<n)-1][n-1]<<endl;
    return 0;
}

/*
状态表示：dp[i][j] 表示所有“经过的点集是i，最后位于点j的路线”的长度的最小值。
状态计算一般对应集合划分。这里可以将dp[i][j]所表示的集合中的所有路线，按倒数第二个点分成若干类，其中第k类是指倒数第二个点是k的所有路线。那么dp[i][j]的值就是每一类的最小值，再取个min。而第k类的最小值就是dp[i - (1 << j)][k] + w[k][j]。
从定义出发，最后dp[(1 << n) - 1][n - 1]就表示所有”经过了所有点，且最后位于点n-1的路线“的长度的最小值
*/
```

### [树形：没有上司从舞会](https://www.acwing.com/problem/content/287/)

```c++
/*
Ural 大学有 N 名职员，编号为 1∼N。
他们的关系就像一棵以校长为根的树，父节点就是子节点的直接上司。
每个职员有一个快乐指数，用整数 Hi 给出，其中 1≤i≤N。
现在要召开一场周年庆宴会，不过，没有职员愿意和直接上司一起参会。
在满足这个条件的前提下，主办方希望邀请一部分职员参会，使得所有参会职员的快乐指数总和最大，求这个最大值。
接下来 N 行，第 i 行表示 i 号职员的快乐指数 Hi。
接下来 N−1 行，每行输入一对整数 L,K，表示 K 是 L 的直接上司。

7
1 1 1 1 1 1 1
1 3
2 3
6 4
7 4
4 5
3 5
res:5
*/

#include<bits/stdc++.h>
using namespace std;
const int N = 6010;

int n,H[N];
int h[N],e[N],ne[N],idx;
bool f[N];   //判断根节点
int dp[N][2];  // 1 表示选 ， 0 表示不选当前节点

void add(int a , int b ){
    e[idx] = b , ne[idx] = h[a] ,h[a] = idx++;
}

void dfs(int u){
    
    dp[u][1] = H[u];    
    for(int i = h[u] ; ~i ; i = ne[i]){
        int j = e[i];
        dfs(j);
        dp[u][1] += dp[j][0];
        dp[u][0] += max(dp[j][0], dp[j][1]);
    }
}
int main(){
    cin>>n;
    for(int i=1;i<=n;i++) cin>>H[i];
    memset(h,-1,sizeof h);
    
    for(int i=1;i<n;i++){
        int a , b;
        cin>>a>>b;  //b 是 a 的上司
        add(b,a);
        f[a] = true;
    }
    
    int root = 1;
    while(f[root]) root++;
    
    dfs(root);
    
    cout<<max(dp[root][0],dp[root][1])<<endl;
    return 0;
}
```

### [记忆化：滑雪](https://www.acwing.com/problem/content/903/)

```c++
/* 给定一个整数矩阵，找出最长递增路径的长度。
对于每个单元格，你可以往上，下，左，右四个方向移动。
输入: nums = 
[
  [9,9,4],
  [6,6,8],
  [2,1,1]
] 
输出: 4 
解释: 最长递增路径为 [1, 2, 6, 9]。
*/

#include <bits/stdc++.h>

using namespace std;
const int N = 3e2+3;

int a[N][N] , n ,m;
int dp[N][N];
int dir[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};

int dfs(int x , int y){
    if(dp[x][y]!=-1) return dp[x][y];
    dp[x][y] = 1;
    for(int i=0;i<4;i++){
        int nx = x+dir[i][0];
        int ny = y+dir[i][1];
        if(nx>=1 && ny>=1 && nx <=n && ny<=m && a[x][y] > a[nx][ny]) {
            dp[x][y] = max(dp[x][y] , dfs(nx,ny)+1);
        }
    }
    return dp[x][y];
}

int main(){
    cin>>n>>m;
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++) cin>>a[i][j];
    }
    memset(dp,-1,sizeof dp);
    int ans = 0;
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            ans = max(dfs(i,j),ans);
        }
    }
    cout<<ans<<endl;
    return 0;
}


```

### [方格取数](https://www.acwing.com/problem/content/1029/)

![image-20210519150647842](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20210519150647842.png)

```C++

/*
某人从图中的左上角A出发，可以向下行走，也可以向右行走，直到到达右下角B的点。
在走过的路上，他可以取走方格中的数（取走后的方格中将变为数字0）。
此人从 A 点到 B 点共走了两次，试找出两条这样的路径，使得取得的数字和为最大。
第一行为一个整数N，表示 N×N 的方格图。
接下来的每行有三个整数，第一个为行号数，第二个为列号数，第三个为在该行、该列上所放的数。
行和列编号从 1 开始。
一行“0 0 0”表示结束。
N≤10
*/

#include<bits/stdc++.h>
using namespace std;

const int N = 12;
int w[N][N],n;
//两个人同时从左上角出发，到达右下角
int f[N][N][N][N];

int main(){
    cin>>n;
    int x,y,z;
    while(cin>>x>>y>>z&&x&&y&&z) w[x][y] = z;
    
    for(int i=1 ; i<=n ;i++){
        for(int j=1 ; j<=n ;j++){
            for(int k=1 ; k<=n ;k++){
                for(int t=1 ; t<=n; t++){
                    int &v = f[i][j][k][t];
                    v = max(v,f[i-1][j][k-1][t]) ;   //上边转移，上边转移过来
                    v = max(v,f[i-1][j][k][t-1]) ;   //上边转移，左边转移过来
                    v = max(v,f[i][j-1][k][t-1]) ;   //左边转移，左边转移过来
                    v = max(v,f[i][j-1][k-1][t]) ;   //左边转移，上边转移过来
                    v+=(w[i][j]+w[k][t]);
                    if(i==k && j == t) v-=w[i][j];    //重复
                }
            }
        }
    }
    
    cout<<f[n][n][n][n] <<endl;   //两个人都走到右下角
    return 0;
}
```

### [导弹防御系统](https://www.acwing.com/problem/content/189/)

```c++
/*为了对抗附近恶意国家的威胁，R 国更新了他们的导弹防御系统。
一套防御系统的导弹拦截高度要么一直 严格单调 上升要么一直 严格单调 下降。
例如，一套系统先后拦截了高度为 3 和高度为 4 的两发导弹，那么接下来该系统就只能拦截高度大于 4 的导弹。
给定即将袭来的一系列导弹的高度，请你求出至少需要多少套防御系统，就可以将它们全部击落。
5
3 5 2 4 1
对于给出样例，最少需要两套防御系统。
一套击落高度为 3,4 的导弹，另一套击落高度为 5,2,1 的导弹。
*/
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 55;

int n;
int h[N];
int up[N], down[N];
int ans;

void dfs(int u, int su, int sd)
{
    if (su + sd >= ans) return;
    if (u == n)
    {
        ans = min(ans, su + sd);
        return;
    }

    int k = 0;
    while (k < su && up[k] >= h[u]) k ++ ;
    if (k < su)
    {
        int t = up[k];
        up[k] = h[u];
        dfs(u + 1, su, sd);
        up[k] = t;
    }
    else
    {
        up[k] = h[u];
        dfs(u + 1, su + 1, sd);
    }

    k = 0;
    while (k < sd && down[k] <= h[u]) k ++ ;
    if (k < sd)
    {
        int t = down[k];
        down[k] = h[u];
        dfs(u + 1, su, sd);
        down[k] = t;
    }
    else
    {
        down[k] = h[u];
        dfs(u + 1, su, sd + 1);
    }
}

int main()
{
    while (cin >> n, n)
    {
        for (int i = 0; i < n; i ++ ) cin >> h[i];
        ans = n;
        dfs(0, 0, 0);
        cout << ans << endl;
    }
    return 0;
}
```

### [最长公共上升子序列](https://www.acwing.com/problem/content/274/)

```c++
#include <bits/stdc++.h>
using namespace std;

const int N = 3010 ;
int f[N][N] ;  //代表所有a[1 ~ i]和b[1 ~ j]中以b[j]结尾的公共上升子序列的集合；
int n;
int a[N],b[N];

int main(){
    cin>>n;
    for(int i=1;i<=n;i++) cin>>a[i];
    for(int i=1;i<=n;i++) cin>>b[i];
    
    for(int i=1;i<=n;i++){
        int maxv = 1;
        for(int j=1;j<=n;j++){
            f[i][j] = f[i-1][j];  // a[i]!=b[j]
            if(a[i] == b[j]) f[i][j] = max(f[i][j] , maxv);
            if(a[i] > b[j]) maxv = max(maxv , f[i-1][j]+1);
        }
    }
    
    int res = 0;
    for(int i=1;i<=n;i++) res = max(f[n][i],res);
    cout<<res<<endl;
    return 0;
}
```

### [状压：两两gcd和最大](https://leetcode-cn.com/problems/maximize-score-after-n-operations/)

```c++
/*给你 nums ，它是一个大小为 2 * n 的正整数数组。你必须对这个数组执行 n 次操作。
在第 i 次操作时（操作编号从 1 开始），你需要：
选择两个元素 x 和 y 。
获得分数 i * gcd(x, y) 。
将 x 和 y 从 nums 中删除。
请你返回 n 次操作后你能获得的分数和最大为多少。
函数 gcd(x, y) 是 x 和 y 的最大公约数。
*/
class Solution {
public:

    int n;
    int f[15][15];            //初始化两两的gcd
    int dp[1<<15];            //保存所有的状态
    int maxScore(vector<int>& nums) {
        n = nums.size();  
        for(int i=0;i<n;i++){
            for(int j=i+1;j<n;j++){
                f[i][j]  = __gcd(nums[i] , nums[j]);
            }
        }
        
        for(int i=0;i<(1<<n);i++){       //枚举所有的状态
            int cnt = lowbit(i);         //看当前状态 1 的个数
            if(cnt&1) continue;          //如果不是偶数个，直接排除
            for(int j=0;j<n;j++){        //枚举第一个数
                for(int k=j+1;k<n;k++){  //枚举第二个数
                    int state = (1<<j) | (1<<k);
                    //假设i：001111
                    //那么满足状态的有6种情况：(j,k):(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
                    if( (state & i) != state) continue;   //必须满足在状态内
                    //cout<<i<<" "<<state<<" "<<j<<" "<<k<<" "<<cnt<<endl;
                    dp[i] = max(dp[i] , dp[i-state] + f[j][k] * (cnt>>1));
                }
            }
        }
        return dp[(1<<n)-1];
    }

    int lowbit(int x){
        int ans = 0;
        while(x){
            ans++;
            x = x&(x-1);
        }
        return ans;
    }
};
```

### [卖卖股票K次](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

```c++
#include <bits/stdc++.h>
using namespace std;
class Solution {
public:
    int maxProfit(int K, vector<int>& prices) {
        int n=prices.size();
        if(n<=1)    return 0;
        //因为一次交易至少涉及两天，所以如果k大于总天数的一半，就直接取天数一半即可，多余的交易次数是无意义的
        K=min(K,n/2);

        /*dp定义：dp[i][j][k]代表 第i天交易了k次时的最大利润，其中j代表当天是否持有股票，0不持有，1持有*/
        int dp[n][2][K+1] ;
        memset(dp , 0 , sizeof dp);
        for(int k=0;k<=K;k++){
            dp[0][0][k]=0;
            dp[0][1][k]=-prices[0];
        }

        /*状态方程：
        dp[i][0][k]，
            当天不持有股票时，看前一天的股票持有情况 ,
        dp[i][1][k]，
            当天持有股票时，看前一天的股票持有情况*/
        for(int i=1;i<n;i++){
            for(int k=1;k<=K;k++){
                //               不持有股票        持有股票卖掉
                dp[i][0][k]=max(dp[i-1][0][k],dp[i-1][1][k]+prices[i]);
                //                持有股票         不持有股票买入
                dp[i][1][k]=max(dp[i-1][1][k],dp[i-1][0][k-1]-prices[i]);
            }
        }
        return dp[n-1][0][K];
    }
};

/*
给定一个整数数组 prices ，它的第 i 个元素 prices[i] 是一支给定的股票在第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

输入：k = 2, prices = [2,4,1]
输出：2

输入：k = 2, prices = [3,2,6,5,0,3]
输出：7
 
提示：
0 <= k <= 100
0 <= prices.length <= 1000
0 <= prices[i] <= 1000
*/
```

### [营救公主](https://leetcode-cn.com/problems/dungeon-game/)

```c++
#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int calculateMinimumHP(vector<vector<int>>& w) {
        int n = w.size() , m = w[0].size();
        //f[i][j] 表示 从 [i,j] 走到终点的最少健康点数
        vector<vector<int>> f (n , vector<int>(m,1e8));

        for(int i = n-1 ; ~i ; i--){
            for(int j=m-1 ; ~j ; j--){
                if(i==n-1 && j == m-1) f[i][j] = max(1 , 1-w[i][j]);
                else{
                    if(i+1 < n) f[i][j] = f[i+1][j] - w[i][j];
                    if(j+1 < m) f[i][j] = min(f[i][j] , f[i][j+1] - w[i][j]);
                    f[i][j] = max(1 , f[i][j]);
                }
            }
        }
        return f[0][0];
    }
};


/*
一些恶魔抓住了公主（P）并将她关在了地下城的右下角。地下城是由 M x N 个房间组成的二维网格。我们英勇的骑士（K）最初被安置在左上角的房间里，他必须穿过地下城并通过对抗恶魔来拯救公主。
骑士的初始健康点数为一个正整数。如果他的健康点数在某一时刻降至 0 或以下，他会立即死亡。
有些房间由恶魔守卫，因此骑士在进入这些房间时会失去健康点数（若房间里的值为负整数，则表示骑士将损失健康点数）；其他房间要么是空的（房间里的值为 0），要么包含增加骑士健康点数的魔法球（若房间里的值为正整数，则表示骑士将增加健康点数）。
为了尽快到达公主，骑士决定每次只向右或向下移动一步。

编写一个函数来计算确保骑士能够拯救到公主所需的最低初始健康点数。
例如，考虑到如下布局的地下城，如果骑士遵循最佳路径 右 -> 右 -> 下 -> 下，则骑士的初始健康点数至少为 7。

-2(K)  -3	 3
-5	   -10	 1
10	   30   -5(P)
 
说明:
骑士的健康点数没有上限。
任何房间都可能对骑士的健康点数造成威胁，也可能增加骑士的健康点数，包括骑士进入的左上角房间以及公主被监禁的右下角房间。
*/
```



# 贪心：

### 贪心：「土」巨石滚滚

```c++

/*
帕秋莉掌握了一种土属性魔法
她使用这种魔法建造了一个大型的土球，并让其一路向下去冲撞障碍
土球有一个稳定性x，如果x < 0，它会立刻散架
每冲撞一个障碍，土球会丧失ai的稳定性，冲撞之后，又会从障碍身上回馈bi的稳定性
帕秋莉想知道，如果合理的安排障碍的顺序，在保证土球不散架的情况下，是否可以将障碍全部撞毁呢？
*/
#include<bits/stdc++.h>
using namespace std;
const int maxn = 5e5+5;
typedef long long ll;
ll n,m;
struct node{
    ll a,b;
    bool operator <(const node &x)
    {
        if(b-a>=0&&x.b-x.a>=0)    return a<x.a;
        if(b-a>=0)   return 1;
        if(x.b-x.a>=0)   return 0;
        return b>x.b;
    }
}x[maxn];


int main(){
    int t;
    cin>>t;
    while(t--){
        cin>>n>>m;
        for(int i=0;i<n;i++){
            cin>>x[i].a>>x[i].b;
        }
        sort(x , x+n);
        //for(int i=0;i<n;i++) cout<<x[i].a<<" "<<x[i].b<<endl;
        int i=0,f = 1;
        for(i=0;i<n;i++){
            m = m-x[i].a;
            if(m<0) {f = 0 ; break;}
            m = m+x[i].b;
        }
        if(f)puts("Yes");
        else puts("No");
    }
}
```

### [贪心：能整除k的最长子数组](https://ac.nowcoder.com/acm/contest/11746/B)

```c++
/*
对于小宝来说，如果一个数组的总和能够整除他的幸运数字k，就是他的幸运数组，而其他数组小宝都很讨厌。现在有一个长度为n的数组，小宝想知道这个数组的子数组中，最长的幸运子数组有多长。
（a-b）%k = a%k-b%k
贪心寻找距离最远的相同的取模结果即可
*/
#include <bits/stdc++.h>
using namespace std;
const int N = 1e5 +10;
int n,k;
map<int,int>mp;
int main(){
	int T=1;
	scanf("%d",&T);
	while(T--){
		scanf("%d%d",&n,&k);
		mp.clear();
		long long  sum=0;
		int ans=-1 , x;
		mp[0]=0;
		for(int i=1;i<=n;i++){
            scanf("%d",&x);
			sum+=x;
			sum%=k;
			if(mp.find(sum)!=mp.end()) ans=max(ans,i-mp[sum]);
			else mp[sum]=i; 
		}
		printf("%d\n",ans);
	}
	return 0;
}
```

# 数学知识

### 科普

```c++
//因子个数和因子之和
如果 N = p1^c1 * p2^c2 * ... *pk^ck
约数个数： (c1 + 1) * (c2 + 1) * ... * (ck + 1)
约数之和： (p1^0 + p1^1 + ... + p1^c1) * ... * (pk^0 + pk^1 + ... + pk^ck)

//欧拉函数
对于正整数n,欧拉函数是小于或等于n的正整数中与n互质的数的数目，记作φ(n), 特：φ(1)=1
φ(n) = n*(c1-1)/c1*(c2-1)/c2*....*(ck-1)/ck
    
//欧拉筛
i % primes[j] == 0时：primes[j]是i的最小质因子，也是primes[j] * i的最小质因子，
因此1 - 1 / primes[j]这一项在phi[i]中计算过了，
只需将基数N修正为primes[j]倍，最终结果为phi[i] * primes[j]。
//质数i的欧拉函数即为phi[i] = i - 1
i % primes[j] != 0：primes[j]不是i的质因子，只是primes[j] * i的最小质因子，
因此不仅需要将基数N修正为primes[j]倍，还需要补上1 - 1 / primes[j]这一项，
因此最终结果phi[i] * (primes[j] - 1)。
  
//逆元
 b存在乘法逆元的充要条件是 b 与模数 m 互质。当模数 m 为质数时，b^(m-2) 即为 b 的乘法逆元。

//等比数列快速幂取模
求等比为k的等比数列之和T[n]
当n为偶数..T[n] = T[n/2] + pow(k,n/2) * T[n/2]
当n为奇数..T[n] = T[n/2] + pow(k,n/2) * T[n/2] + pow(k,n)
```

### 欧拉函数

```c++
int primes[N], cnt;     // primes[]存储所有素数
int phi[N];           // 存储每个数的欧拉函数
bool st[N];             // st[x]存储x是否被筛掉

void get_phis(int n){
    phi[1] = 1;
    for (int i = 2; i <= n; i ++ ){
        if (!st[i]){
            primes[cnt ++ ] = i;
            phi[i] = i - 1;
        }
        for (int j = 0; primes[j] <= n / i; j ++ ){
            int t = primes[j] * i;
            st[t] = true;
            if (i % primes[j] == 0){
                phi[t] = phi[i] * primes[j];
                break;
            }
            phi[t] = phi[i] * (primes[j] - 1);
        }
    }
}
```

### 扩展欧几里得

```c++
// 求x, y，使得ax + by = gcd(a, b)
int exgcd(int a, int b, int &x, int &y)
{
    if (!b)
    {
        x = 1; y = 0;
        return a;
    }
    int d = exgcd(b, a % b, y, x);
    y -= (a/b) * x;
    return d;
}
```

### [分解质因数：最小指数](https://ac.nowcoder.com/acm/contest/9510/B)

```c++
#include<bits/stdc++.h>
using namespace std;
#define ll long long
const int maxn=1e4+10,INF=0x3f3f3f3f;
ll primer[maxn],idx;
bool vis[maxn];
int tot;
void init(){
    for(ll i=2; i<maxn; i++){
    	if (vis[i]) continue;
    	primer[++idx]=i;
    	for (ll j=i*i; j<maxn; j+=i) {
    		vis[j]=true;
    	}
    }
}
int main(){
    init();
    int t; 
	scanf("%d",&t);
    while(t--){
        ll n;
        scanf("%lld",&n);
        int ans=INF;
        for(int i=1; i<=idx; i++){
            if(primer[i]>n) break;
            int x=0;
            while(n%primer[i]==0){
                n/=primer[i];
                x++;
            }
            if(x) ans=min(ans,x);
        }
        if(n==1||ans==1) printf("%d\n",ans==INF?0:ans);
        else {
            ll m1=(ll)sqrt(sqrt(n));
            ll m2=(ll)sqrt(n);
            ll m3=(ll)pow(n*1.0,1.0/3.0);
            if(m1*m1*m1*m1==n) ans=min(ans,4);
            else if(m3*m3*m3==n||(m3-1)*(m3-1)*(m3-1)==n||(m3+1)*(m3+1)*(m3+1)==n) ans=min(ans,3);
            else if(m2*m2==n) ans=min(ans,2);
            else ans=1;
            printf("%d\n",ans);
        }
    }
}
```



### [区间因子个数和](https://ac.nowcoder.com/acm/contest/7817/E)

```c++
#include<iostream>
#include<cmath>
using namespace std;
typedef long long ll;
ll cal(ll k){
	ll ans = 0 ;
	ll t = sqrt((double)k);
	for(ll i=1;i<=t;i++){
		ans = ans+ k/i;
	}
	return ans*2 -t*t;
}
int main(){
    ll n , m ;
    cin>>n>>m;
    cout<<cal(m)-cal(n-1)<<endl;
}

```

### 矩阵快速幂

```c++
#include <iostream>
#include <cstddef>
#include <cstring>
#include <vector>
using namespace std;
typedef long long ll;
const int mod=10000;
typedef vector<ll> vec;
typedef vector<vec> mat;
mat mul(mat &a,mat &b)
{
    mat c(a.size(),vec(b[0].size()));
    for(int i=0; i<2; i++)
    {
        for(int j=0; j<2; j++)
        {
            for(int k=0; k<2; k++)
            {
                c[i][j]+=a[i][k]*b[k][j];
                c[i][j]%=mod;
            }
        }
    }
    return c;
}
mat pow(mat a,ll n)
{
    mat res(a.size(),vec(a.size()));
    for(int i=0; i<a.size(); i++)
        res[i][i]=1;//单位矩阵；
    while(n)
    {
        if(n&1) res=mul(res,a);
        a=mul(a,a);
        n/=2;
    }
    return res;
}
ll solve(ll n)
{
    mat a(2,vec(2));
    a[0][0]=1;
    a[0][1]=1;
    a[1][0]=1;
    a[1][1]=0;
    a=pow(a,n);
    return a[0][1];//也可以是a[1][0];
}
int main()
{
    ll n;
    while(cin>>n&&n!=-1)
    {
        cout<<solve(n)<<endl;
    }
    return 0;
}
```



### [01分数规划](https://ac.nowcoder.com/acm/problem/14662)

```c++
/*
n个手办，想买k个手办，买下来的东西的总价值/总花费=max。
第一行一个整数T，为数据组数。
对于每组数据，第一行两个正整数n，k，如题。
接下来n行，每行有两个正整数ci，vi。分别为手办的花费和它对于小咪的价值。
对于每组数据，输出一个数，即能得到的总价值/总花费的最大值。精确至整数。
*/
#include<bits/stdc++.h>
using namespace std;
const int maxn = 1e4+10;
typedef long long ll;
int t,n,k;
struct node{
    int c,v;
}a[maxn];
double w[maxn];
bool judg(int x)
{
    double sum=0;
    for(int i=0;i<n;++i) w[i]=a[i].v-a[i].c*x;
    //v1+v2+v3+..+vn = x(c1+c2+c3+..+cn)
    sort(w,w+n,greater<double>());
    for(int i=0;i<k;++i) sum+=w[i];
    return sum<0;
}
int main()
{

    cin>>t;
    while(t--)
    {
        cin>>n>>k;
        for(int i=0;i<n;++i) cin>>a[i].c>>a[i].v;
        int l=0 , r=maxn , mid;
        while(l<r)
        {
            mid=l+r+1>>1;
            if(judg(mid)) r=mid-1;
            else l=mid;
        }
        cout<<r<<endl;
    }
    return 0;
}
```

### [两圆的关系](https://ac.nowcoder.com/acm/contest/8564/H)

```c++
#include<bits/stdc++.h>
using namespace std;
class Point {
	private:
	double m_x,m_y;
	public:
	void set(double x, double y) {
		m_x = x; m_y = y;
	}
	double dispoint(Point& another) {
	return hypot(m_x - another.m_x, m_y - another.m_y);
}
};
class Circle {
	private:
	double m_r;
	Point p;
	public:
	void set(double x, double y) {
		p.set(x, y);
	}
	void setr(double r) {
		m_r = r;
	}
	int judge(Circle &another) {
		double rr = m_r + another.m_r;
		double rs = fabs(m_r - another.m_r);
		double dis = p.dispoint(another.p);
		if (rr == dis) return 1; //外切
		else if (rr < dis) return 2; //外离
		else if (dis<rr && dis>rs) return 3;//相交
		else if (rs == dis) return 4;//内切
		else if (rs > dis) return 5;//内含
	}
};
int main() {
	Circle c1, c2;
	double x, y,r,jude;
	int t;
	cin >> t;
	while(t--)
	{
		cin >> x >> y >>r;
		c1.set(x, y);
		c1.setr(r);
		cin >> x >> y >> r;
		c2.set(x, y);
		c2.setr(r);
		jude = c1.judge(c2);
		if(jude==1||jude==3||jude==4)cout << "YES" << endl;  // 圆弧相交
		else cout << "NO" << endl;
	}
	return 0;	
}

```

### 【L,R】各位之和

```c++
#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
LL sum(LL n)
{
    LL ans = 0, x;
    while(n) {
        x = n % 10;
        n /= 10;
        ans += ((1 + x) * x) / 2 + n * 45;
    }
    return ans;
}

int main()
{
    LL a, b;
    while(cin>>a>>b){
    	if(a<0 || b<0) break;
    	cout<<sum(b)-sum(a-1)<<endl;
	}
    return 0;
}

```

### 最大亦或对

```c++
//在给定的 N 个整数 A1，A2……AN 中选出两个进行 xor（异或）运算，得到的结果最大是多少？
#include<iostream>
#include<algorithm>

using namespace std;
const int N = 1e5 + 10, M = 3e6 + 10;//M=31*N
int a[N], son[M][2], idx;

void insert(int x) {
    int p = 0;
    for (int i = 30; i >= 0; i--) {
        int s = x >> i & 1;
        if (!son[p][s]) son[p][s] = ++idx;
        p = son[p][s];
    }
}

int query(int x) {
    int p = 0;
    int res = 0;
    for (int i = 30; i >= 0; i--) {
        int s = x >> i & 1;
        if (son[p][!s]) {
            res += 1 << i;
            p = son[p][!s];
        } else p = son[p][s];
    }
    return res;
}

int main() {
    int n;
    cin >> n;
    for (int i = 0; i < n; i++) {
        scanf("%d", &a[i]);
        insert(a[i]);
    }
    int res = 0;
    for (int i = 0; i < n; i++) {
        res = max(res, query(a[i]));
    }
    cout << res;
    return 0;
}

```

### [直线上最多的点数](https://leetcode-cn.com/problems/max-points-on-a-line/)

```c++
class Solution {
public:
    int maxPoints(vector<vector<int>>& points) {
        typedef long double LD;

        int res = 0;
        for (auto& p: points) {    //枚举每一个点作为定点
            int ss = 0, vs = 0;   // ss 表示与定点重合 ， vs表示与定点垂直，没斜率，单独算
            unordered_map<LD, int> cnt;   //哈希表存直线
            for (auto& q: points)       //其他点与定点构成的直线 ， 斜率相同就是同一条直线
                if (p == q) ss ++ ;
                else if (p[0] == q[0]) vs ++ ;
                else {
                    LD k = (LD)(q[1] - p[1]) / (q[0] - p[0]);
                    cnt[k] ++ ;
                }
            int c = vs;
            for (auto [k, t]: cnt) c = max(c, t);
            //答案就是每条直线上的点 + 重点 ，
            res = max(res, c + ss);
        }
        return res;
    }
};

/*
给定一个二维平面，平面上有 n 个点，求最多有多少个点在同一条直线上。

示例 1:

输入: [[1,1],[2,2],[3,3]]
输出: 3
解释:
^
|
|        o
|     o
|  o  
+------------->
0  1  2  3  4
示例 2:

输入: [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
输出: 4
解释:
^
|
|  o
|     o        o
|        o
|  o        o
+------------------->
0  1  2  3  4  5  6

*/
```



### 威佐夫博弈

```c++
//有两堆各若干个物品，两个人轮流从某一堆或同时从两堆取同样多的物品，规定每次至少取一个，多者不限，最后取光者获胜。
#include<bits/stdc++.h>
using namespace std;
int main(){
    int a,b,c;
    while(cin>>a>>b){
        if(a>b) swap(a,b);
        //黄金分割，（Betty定理）
        c = floor((b-a)*(sqrt(5.0)+1)/2) ;
        if(a==c) puts("0");
        else puts("1");
    }
}

```

### 进制转换

```java
import java.math.BigInteger;
import java.util.*;
public class Main{    
    public static  void main(String[] args)  {  
        Scanner cin = new Scanner(System.in);            
            String str = cin.next();
            int a = cin.nextInt();
            int b = cin.nextInt();
            //A进制的str转换为B进制
            String ans = new BigInteger(str,a).toString(b);
            System.out.println(ans);

        
			string = Integer.toString(a, x);    //把int型数据转换乘X进制数并转换成string型
			System.out.println(string);
			int b = Integer.parseInt(string, x);//把字符串当作X进制数转换成int型
			System.out.println(b);

    }
}


// 二进制加法
def addBinary(self, a: str, b: str) -> str:
	return bin(int(a,2)+int(b,2))[2:]

```

### 康托展开

```c++
#include<bits/stdc++.h>
typedef long long ll;
using namespace std;

const int Fmax = 12 ;
ll fac[Fmax];   //第i位的阶乘 
int ans1 = 0 ;
char ans2[Fmax];

void init(){ 
	fac[0] = fac[1] = 1;
	for(int i=2;i<Fmax;i++){
		fac[i] = i*fac[i-1];
	}
}
// 求这个数在全排列中排在第几 
int kangtuo(string a){ 
	int len = a.length();
	for(int i=0;i<len;i++){
		int count = 0;
		for(int j=i+1;j<len;j++){
			// 计算比当前数小的个数
			if(a[i]>a[j]) count++;
		}
		ans1 = ans1+count*fac[len-i-1];
	}
	return ans1+1;
}
/*比如2143 这个数，求其展开：从头判断，至尾结束,
①比 2（第一位数）小的数有多少个->1个 就是1，1*3!
②比 1（第二位数）小的数有多少个->0个 0*2!
③比 4（第三位数）小的数有多少个->3个就是1,2,3，
                   但是1,2之前已经出现，所以是  1*1!
将所有乘积相加=7 ,比该数小的数有7个，所以该数排第8的位置。*/


//康托展开的逆运算,{1...n}的全排列，中的第k个数为s[]
void reverse_kangtuo(int n,int k)
{
    int i, j, t, vst[Fmax]={0};
    --k;
    for (i=0; i<n; i++)
    {
        t = k/fac[n-i-1];
        for (j=1; j<=n; j++)
            if (!vst[j])
            {
                if (t == 0) break;
                --t;
            }
        ans2[i] = '0'+j;
        vst[j] = 1;
        k %= fac[n-i-1];
    }
}
/*假设求4位数中第19个位置的数字。
① 19减去1  → 18
② 18 对3！作除法 → 得3余0
③  0对2！作除法 → 得0余0
④  0对1！作除法 → 得0余0
据上面的可知：
我们第一位数（最左面的数），比第一位数小的数有3个，显然 第一位数为→ 4
比第二位数小的数字有0个，所以 第二位数为→1
比第三位数小的数字有0个，因为1已经用过，所以第三位数为→2
第四位数剩下 3
该数字为  4123  (正解)
*/ 
int main(){
	init();
	string n;
	cin>>n;
	cout<<kangtuo(n)<<endl;
	
	int m,k;
	cin>>m>>k;
	reverse_kangtuo(m,k);
	cout<<ans2<<endl;
}
```

### [分块：教主的魔法](https://www.luogu.com.cn/problem/P2801)

```c++
/*
每个人的身高一开始都是不超过 1000 的正整数。教主的魔法每次可以把闭区间 [L,R]内的英雄的身高全部加上一个整数 W。（虽然 L=R 时并不符合区间的书写规范，但我们可以认为是单独增加第 L(R) 个英雄的身高）
CYZ、光哥和 ZJQ 等人不信教主的邪，于是他们有时候会问 WD 闭区间 [L, R]内有多少英雄身高大于等于 C，以验证教主的魔法是否真的有效。
WD 巨懒，于是他把这个回答的任务交给了你。
*/

#include<bits/stdc++.h>
using namespace std;
const int maxn=1000010;
const int _maxn=1010;
int n,q,block,cnt;
int belong[maxn],val[maxn],mark[_maxn];//数组在上文已提及，不多赘述 
vector<int> kuai[_maxn];
void start()//初始化kuai数组，让每个块内都有从小到大的顺序 （当然你也可以从大到小qwq） 
{
	for(int i=1;i<=cnt;++i)
		sort(kuai[i].begin(),kuai[i].end());
}
void update(int pos)//更新kuai数组
{
	kuai[pos].clear();//清空数组 
	for(int i=(pos-1)*block+1;i<=pos*block;++i)
		kuai[pos].push_back(val[i]);//因为块内元素已修改，所以重新放入元素 
	sort(kuai[pos].begin(),kuai[pos].end());//保证块内的有序性 
} 
void add(int l,int r,int x)//区间修改 
{
	for(int i=l;i<=min(r,belong[l]*block);++i)
		val[i]+=x;//左边零散的块暴力修改 
	update(belong[l]);//更新左边零散块的kuai数组 
	if(belong[l]!=belong[r])
	{
		for(int i=belong[l]+1;i<belong[r];++i)
			mark[i]+=x;//中间的块直接打标记 
		for(int i=(belong[r]-1)*block+1;i<=r;++i)
			val[i]+=x;//右边零散的块暴力修改		
		update(belong[r]);// 更新右边零散块的kuai数组 
	}
}
int search(int l,int r,int x)
{
	int num=0;//计数器 
	for(int i=l;i<=min(r,belong[l]*block);++i)
		if(val[i]+mark[belong[i]]>=x)//左边零散的块暴力查询，记住要加mark值 
			++num;
	if(belong[l]!=belong[r])
	{
		for(int i=belong[l]+1;i<belong[r];++i)
		{
			int L=0,R=block-1,mid;
			while(L<=R)//中间块二分查询，不然会T ，应该也可以用lower_bound这些类似的函数，别看我。。我不会。。 
			{
				mid=L+((R-L)>>1);
				if(kuai[i][mid]+mark[i]<x)
					L=mid+1;
				else
					R=mid-1;
			}
			num+=block-L;
		}
		for(int i=(belong[r]-1)*block+1;i<=r;++i)
			if(val[i]+mark[belong[i]]>=x)//右边零散的块暴力查询 
				++num;
	}
	return num;//返回值。。。 
}
int main()
{
	scanf("%d%d",&n,&q);
	block=sqrt(n);//块的大小 
	for(int i=1;i<=n;++i)
	{
		scanf("%d",&val[i]);
		if(i%block==1)//到了新的一个块 
			cnt++;
		belong[i]=cnt;
		kuai[cnt].push_back(val[i]);
	}
	start();
	for(int i=1;i<=q;++i)
	{
		char in1;
		int in2,in3,in4;
		scanf(" %c%d%d%d",&in1,&in2,&in3,&in4);
		switch(in1)
		{
			case 'M':				
				add(in2,in3,in4);			
			break;
			case 'A':
				printf("%d\n",search(in2,in3,in4));
			break;
		}
	}
	return 0;
}
```

### [数字 1 的个数](https://leetcode-cn.com/problems/number-of-digit-one/)

```c++
#include <bits/stdc++.h>
using namespace std;
class Solution {
public:
    int countDigitOne(int n) {
        if (n < 1) {
            return 0;
        }

        long digit = 1;
        int high = n / 10, cur = n % 10, low = 0;
        int res = 0;

        while (high != 0 || cur != 0) {
            if (cur == 0) {
                res += high * digit;
            }
            else if (cur == 1) {
                res += high * digit + low + 1;
            }
            else {
                res += (high + 1) * digit;
            }
            low += cur * digit;
            cur = high % 10;
            high /= 10;
            digit *= 10; 
        }

        return res;
    }
};

/*
给定一个整数 n，计算所有小于等于 n 的非负整数中数字 1 出现的个数。
示例 1：

输入：n = 13
输出：6
示例 2：

输入：n = 0
输出：0
 
提示：
0 <= n <= 2 * 10^9
*/

```



# 树状数组

### [单点查询](https://www.acwing.com/problem/content/248/)

```c++
#include<bits/stdc++.h>

using namespace std;
typedef long long LL;
const int N = 1e5+5;

int n, m;
int a[N];
LL tr[N];

void add(int x, int c){
    for (int i = x; i <= n; i += i&-i) tr[i] += c;
}

LL sum(int x){
    LL res = 0;
    for (int i = x; i; i -= i&-i) res += tr[i];
    return res;
}

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &a[i]);
    for (int i = 1; i <= n; i ++ ) add(i, a[i] - a[i - 1]);

    while (m -- )
    {
        char op[2];
        int l, r, d;
        scanf("%s%d", op, &l);
        if (*op == 'C')         //表示把数列中第 l∼r 个数都加 d。
        {
            scanf("%d%d", &r, &d);
            add(l, d);
            add(r + 1, -d);
        }
        else printf("%lld\n", sum(l));   //表示询问数列中第 l 个数的值。
    }

    return 0;
}

```

### [区间查询](https://www.acwing.com/problem/content/244/)

```c++
#include<iostream>
#include<algorithm>
#include<cstdio>

using namespace std;
const int N = 1e5+5;
typedef long long ll;

ll tr1[N] , tr2[N];
int a[N];
int n,m;
char op;
int x,y,k;

int lowbit(int x){
    return x&-x;
}

void add(int x,int val){
    for(int i=x;i<=n;i+=lowbit(i)) tr1[i] += val , tr2[i]+=(ll)val*x;
}

ll sum(int x){
    ll res = 0;
    for(int i=x;i;i-=lowbit(i)) res+=(ll)(x+1)*tr1[i]-tr2[i];
    return res;
}

int main(){
    scanf("%d %d",&n,&m);
    
    for(int i=1;i<=n;i++) scanf("%d",&a[i]);
    for(int i=1;i<=n;i++) {
        int d =  a[i]-a[i-1];
        add(i,d);
    }
    
    while(m--){
		cin>>op;
		if(op=='C'){  
			scanf("%d %d %d",&x,&y,&k); //表示把数列中第 x∼y 个数都加 d。
			add(x,k);
			add(y+1,-k);
		}else{
			scanf("%d %d",&x,&y);
			printf("%lld\n",sum(y)-sum(x-1)); //查询x-y的和
		}
	}
	return 0;
}
```



### [区间真子集](https://ac.nowcoder.com/acm/problem/107078)

```c++
//就是给出N个区间，问这个区间是多少个区间的真子集。
#include<algorithm> 
#include<iostream>
#include<cstring>
#include<cstdio>
using namespace std;
const int maxn = 1e5+5;
int d[maxn] , ans[maxn];
struct node{
	int s , e ,pos;
	bool operator<(const node& t) const {
		// s升序,e降序, 保证前面的牛都一定不比后面的牛弱
        if(t.e == e) return s < t.s;
        return e > t.e;
    }
}st[maxn];
int n , mx; 
void add(int p){
	for(;p<=mx;p+=p&-p) d[p]+=1;
}
int sum(int p){
	int res = 0;
	for(;-p;p-=p&-p) res+=d[p];
	return res;
}
int main(){
	while(~scanf("%d",&n) && n){
		for(int i=0;i<n;i++) {
			scanf("%d%d",&st[i].s,&st[i].e);
			++st[i].s , ++st[i].e;  //防止输入0 
			st[i].pos = i;
			mx = max(mx,st[i].s);
		}
		sort(st , st+n);
		for(int i =0;i<n;i++){
			if(i!=0&&st[i].s==st[i-1].s && st[i].e==st[i-1].e)
			ans[st[i].pos] = ans[st[i-1].pos];   //区间重合不需要重复计算 
			else ans[st[i].pos] = sum(st[i].s);
			add(st[i].s);
		}
		for(int i=0;i<n;i++) {   //输出该区间有多少子区间 
			if(i!=n-1) printf("%d ",ans[i]);
			else printf("%d\n",ans[i]);
		}
		mx = 0;
		memset(ans,0,sizeof(ans));
		memset(d,0,sizeof(d));
	}
	return 0;
}
```

### [二维树状数组](https://ac.nowcoder.com/acm/problem/105792)

```c++
#include<iostream>
#include<cstdio> 
using namespace std;
const int maxn = 1050;
int c[maxn][maxn];
int id,n,x,y,val,xx,yy;

void add(int x , int y ,int val){
	while(x<=n){
		int p = y;
		while(p<=n)c[x][p]+=val,p+=p&-p;
		x+=x&-x;
	}
}

int ask(int x , int y ){
	int res = 0;
	while(x>=1){
		int p = y;
		while(p>=1)res+=c[x][p],p-=p&-p;
		x-=x&-x;
	}
	return res;
}

int main(){
    //输入一个n，表示n*n的二维数组(初始0)
	scanf("%d%d",&id,&n);
	while(1){
		scanf("%d",&id);
		if(id==3) break;
		if(id==1){
			scanf("%d%d%d",&x,&y,&val);
			x++;y++;
            //(x,y)加val
			add(x,y,val);
		}else if(id==2){
            //求(x,y)到(xx,yy)的和
			scanf("%d%d%d%d",&x,&y,&xx,&yy);
			x++;y++;xx++;yy++;
			//二维前缀和 
			int ans = ask(xx,yy)+ask(x-1,y-1)-ask(x-1,yy)-ask(xx,y-1);
			printf("%d\n",ans);
		}
	}
} 
```

# 线段树

### [最大子段乘积](https://leetcode-cn.com/problems/maximum-product-subarray/)

```c++
#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int ans = nums[0] , f = nums[0] , g = nums[0];
        /*
        f[i]表示所有从0到i并且选用nums[i]获得的最大乘积
        g[i]表示所有从0到i并且选用nums[i]获得的最小乘积
        1、
        当nums[i] >= 0时，f[i] = max(nums[i], f[i - 1] * nums[i])
        当nums[i] < 0时，f[i] = max(nums[i], g[i - 1] * nums[i])

        2、
        当nums[i] >= 0时，g[i] = min(nums[i], g[i - 1] * nums[i])
        当nums[i] < 0时，g[i] = min(nums[i], f[i - 1] * nums[i])

        可以通过用滚动数组的方式，记录 f = f[i - 1]， g = g[i - 1]
        */

        for(int i = 1 ; i < nums.size() ; i++){
            int pref = f , preg = g;
            if(nums[i] > 0) {
                f = max(nums[i] , f * nums[i]);
                g = min(nums[i] , g * nums[i]);
            }else{
                f = max(nums[i] , preg * nums[i]);
                g = min(nums[i] , pref * nums[i]);
            }
            ans = max(ans , f);
        }
        return ans;
    }
};

/*
给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
示例 1:
输入: [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
示例 2:
输入: [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
*/
```



### [最大子段和](https://www.acwing.com/problem/content/246/)

```c++
#include<bits/stdc++.h>
using namespace std;

const int N = 500010;

int a[N];
int op,x,y,n,m;
struct Node{
    int l,r;
    int sum,lsum,rsum,tsum;
    // sum:区间和   lsum:最大前缀和  rsum:最大后缀和 tsum:最大连续子段和
}st[N*4];


void pushup(Node &u , Node &l , Node &r){
    u.sum = l.sum + r.sum;
    u.lsum = max(l.lsum , l.sum+r.lsum);
    u.rsum = max(r.rsum , r.sum+l.rsum);
    u.tsum = max(max(l.tsum , r.tsum),l.rsum+r.lsum);
}

void pushup(int u){
    pushup( st[u] , st[u<<1] , st[u<<1|1] );
}

void build(int u , int l , int r){
    if(l == r) st[u] = { l,r,a[l],a[l],a[l],a[l] } ;
    else{
        st[u] = {l,r};
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}

void modify(int u , int x , int v){
    if(st[u].l == x && st[u].r == x) st[u] = {x,x,v,v,v,v};
    else {
        int mid = st[u].l + st[u].r >> 1 ;
        if(x<=mid) modify(u<<1 , x , v);
        else  modify(u<<1|1 , x, v);
        pushup(u);
    }
}

Node query(int u , int l , int r){
    if(st[u].l >= l && st[u].r <= r) return st[u];
    else {
        int mid = st[u].l + st[u].r  >> 1;
        if(r<= mid) return query(u<<1 , l , r);     // 全部在左
        else if (l > mid) return query(u << 1|1, l, r);  //全部在右
        else {      //中间
            auto left = query(u << 1, l, r);
            auto right = query(u << 1|1, l, r);
            Node res;
            pushup(res, left, right);
            return res;
        }
    }
}

int main(){
    
    scanf("%d %d", &n, &m);
    for(int i=1;i<=n;i++) scanf("%d",&a[i]);
    build(1,1,n);
    
    while(m--){
        scanf("%d%d%d",&op,&x,&y);
        if(op==1){    //查询区间 [x,y] 中的最大连续子段和
            if(x>y) swap(x,y);
            printf("%d\n",query(1,x,y).tsum);
        }else modify(1,x,y); //把 A[x] 改成 y
    }
    return 0;
}
```

### [区间最大公约数](https://www.acwing.com/problem/content/247/)

```c++
#include<bits/stdc++.h>
using namespace std;

typedef long long LL;
const int N = 500010;

int n, m;
LL w[N];
struct Node
{
    int l, r;
    LL sum, d;
}tr[N * 4];

LL gcd(LL a, LL b){return b ? gcd(b, a % b) : a;}

void pushup(Node &u, Node &l, Node &r)
{
    u.sum = l.sum + r.sum;
    u.d = gcd(l.d, r.d);
}

void pushup(int u)
{
    pushup(tr[u], tr[u << 1], tr[u << 1 | 1]);
}

void build(int u, int l, int r)
{
    if (l == r)
    {
        LL b = w[r] - w[r - 1];
        tr[u] = {l, r, b, b};
    }
    else
    {
        tr[u].l = l, tr[u].r = r;
        int mid = l + r >> 1;
        build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}

void modify(int u, int x, LL v)
{
    if (tr[u].l == x && tr[u].r == x)
    {
        LL b = tr[u].sum + v;
        tr[u] = {x, x, b, b};
    }
    else
    {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid) modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}

Node query(int u, int l, int r)
{
    if (tr[u].l >= l && tr[u].r <= r) return tr[u];
    else
    {
        int mid = tr[u].l + tr[u].r >> 1;
        if (r <= mid) return query(u << 1, l, r);
        else if (l > mid) return query(u << 1 | 1, l, r);
        else
        {
            auto left = query(u << 1, l, r);
            auto right = query(u << 1 | 1, l, r);
            Node res;
            pushup(res, left, right);
            return res;
        }
    }
}

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++ ) scanf("%lld", &w[i]);
    build(1, 1, n);

    int l, r;
    LL d;
    char op[2];
    while (m -- )
    {
        scanf("%s%d%d", op, &l, &r);
        if (*op == 'Q'){
            auto left = query(1, 1, l);     // [l,r]的最大gcd
            Node right({0, 0, 0, 0});
            if (l + 1 <= r) right = query(1, l + 1, r);
            printf("%lld\n", abs(gcd(left.sum, right.d)));
        }
        else{
            scanf("%lld", &d);     //[l,r]每个数都加上d
            modify(1, l, d);
            if (r + 1 <= n) modify(1, r + 1, -d);
        }
    }

    return 0;
}
```



### st表

```c++
#include<bits/stdc++.h>
using namespace std;
const int maxn=1e6+10;
int n,m;
int a[maxn];
int st[maxn][20];
void init()
{
	int len=log(n)/log(2)+1;
	for(int i=1;i<=n;i++) st[i][0]=a[i];
	for(int j=1;j<len;j++)
	{
		for(int i=1;i<=n-(1<<j)+1;i++)
		{
			st[i][j]=min(st[i][j-1],st[i+(1<<j-1)][j-1]);
		}
	}	
}
int f(int l,int r)
{
	int k=log(r-l+1)/log(2);
	return min(st[l][k],st[r-(1<<k)+1][k]);
}
```

### [区间修改+](https://www.acwing.com/problem/content/244/)

```c++
#include<bits/stdc++.h>
using namespace std;

const int N = 1e5+5;
typedef long long ll;
int a[N];
int d,l,r,n,m;
char op;

struct node{
    int l,r,lazy;
    ll sum;
}st[N*4];

void pushup(int u){
    st[u].sum = st[u<<1].sum + st[u<<1|1].sum;
}

void pushdown(int u){
    auto &root = st[u], &left = st[u << 1], &right = st[u << 1 | 1];
    left.lazy += root.lazy; 
    left.sum += (ll)(left.r - left.l + 1) * root.lazy;
    right.lazy += root.lazy; 
    right.sum += (ll)(right.r - right.l + 1) * root.lazy;
    root.lazy = 0;
}

void build(int u , int l, int r){
    if(l==r) st[u] = {l,r,0,a[l]};
    else {
        int mid = l+r>>1;
        st[u] = {l,r,0,0};
        build(u<<1,l,mid);
        build(u<<1|1,mid+1,r);
        pushup(u);
    }
}

ll query(int u , int l , int r){
    if(st[u].l >=l && st[u].r <= r) return st[u].sum;
    else {
        pushdown(u);
        ll res = 0;
        int mid = st[u].l + st[u].r >> 1 ;
        if(l<=mid) res+=query(u<<1,l,r);
        if(r>mid) res+=query(u<<1|1,l,r);
        return res;
    }
}

void modify(int u , int l , int r , int d){
    if(st[u].l >= l && st[u].r <= r){
        st[u].sum += (ll)(st[u].r - st[u].l + 1) * d;
        st[u].lazy += d;
    }
    else {
        pushdown(u);
        int mid = st[u].l + st[u].r >> 1 ;
        if(l<=mid) modify(u<<1,l,r,d);
        if(r>mid) modify(u<<1|1,l,r,d);
        pushup(u);
    }
}

int main(){
    scanf("%d%d", &n, &m);
    for(int i=1;i<=n;i++) scanf("%d",&a[i]);
    build(1,1,n);
    
    while(m--){
        scanf(" %c",&op);
        if(op=='C'){
            scanf("%d %d %d",&l,&r,&d);   //数列 l-r 都加上 d
            modify(1,l,r,d);
        }else if(op=='Q'){
            scanf("%d %d",&l,&r);     // l - r 的和
            printf("%lld\n",query(1,l,r));
        }
    }
    return  0;
}
```

### [区间修改*+](https://www.acwing.com/problem/content/1279/)

```c++
#include<bits/stdc++.h>

using namespace std;

typedef long long LL;

const int N = 100010;

int n, p, m;
int w[N];
struct Node
{
    int l, r;
    int sum, add, mul;
}tr[N * 4];

void pushup(int u)
{
    tr[u].sum = (tr[u << 1].sum + tr[u << 1 | 1].sum) % p;
}

void eval(Node &t, int add, int mul)
{
    t.sum = ((LL)t.sum * mul + (LL)(t.r - t.l + 1) * add) % p;
    t.mul = (LL)t.mul * mul % p;
    t.add = ((LL)t.add * mul + add) % p;
}

void pushdown(int u)
{
    eval(tr[u << 1], tr[u].add, tr[u].mul);
    eval(tr[u << 1 | 1], tr[u].add, tr[u].mul);
    tr[u].add = 0;
    tr[u].mul = 1;
}

void build(int u, int l, int r)
{
    if (l == r) tr[u] = {l, r, w[r], 0, 1};
    else
    {
        tr[u] = {l, r, 0, 0, 1};
        int mid = l + r >> 1;
        build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}

void modify(int u, int l, int r, int add, int mul)
{
    if (tr[u].l >= l && tr[u].r <= r) eval(tr[u], add, mul);
    else
    {
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid) modify(u << 1, l, r, add, mul);
        if (r > mid) modify(u << 1 | 1, l, r, add, mul);
        pushup(u);
    }
}

int query(int u, int l, int r)
{
    if (tr[u].l >= l && tr[u].r <= r) return tr[u].sum;

    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    int sum = 0;
    if (l <= mid) sum = query(u << 1, l, r);
    if (r > mid) sum = (sum + query(u << 1 | 1, l, r)) % p;
    return sum;
}

int main()
{
    scanf("%d%d", &n, &p);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &w[i]);
    build(1, 1, n);

    scanf("%d", &m);
    while (m -- )
    {
        int t, l, r, d;
        scanf("%d%d%d", &t, &l, &r);
        if (t == 1)    //数列中的一段数全部乘一个值
        {
            scanf("%d", &d);
            modify(1, l, r, 0, d);
        }
        else if (t == 2)   //数列中的一段数全部加一个值
        {
            scanf("%d", &d);
            modify(1, l, r, d, 1);
        }
        else printf("%d\n", query(1, l, r));  //数列中的一段数的和%p
    }

    return 0;
}
```

### [第八大的数](https://www.acwing.com/problem/content/description/2558/)

```c++
/*输入的第一行包含两个整数 L,N，分别表示河流的长度和要你处理的信息的数量。开始时河流沿岸没有建筑，或者说所有的奇特值为 0。
接下来 N 行，每行一条你要处理的信息。
如果信息为 C p x，表示流域中第 p 个位置 (1≤p≤L) 建立了一个建筑，其奇特值为 x。如果这个位置原来有建筑，原来的建筑会被拆除。
如果信息为 Q a b，表示有个人生活的范围是河流的第 a 到 b 个位置（包含 a 和 b，a≤b），这时你要算出这个区间的第八大奇迹的奇特值，并输出。如果找不到第八大奇迹，输出 0。
输出格式
对于每个为 Q 的信息，你需要输出一个整数，表示区间中第八大奇迹的奇特值。
*/
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;
const int N = 100010 ;
const int M = 8;
struct node{
    int l , r ;
    vector<int> v;
}st[N*4];

int n , m , p , x , a[N];
char op;

void pushup(vector<int> &v, const vector<int> &l, const vector<int> &r)
{
    static int i, j, k; i = j = k = 0;
    while (i < M && j < M && k < M)
        if (l[i] > r[j]) v[k ++ ] = l[i ++ ];
        else    v[k ++ ] = r[j ++ ];
    while (i < M && k < M) v[k ++ ] = l[i ++ ];
    while (j < M && k < M) v[k ++ ] = r[j ++ ];
}

void build(int u, int l, int r)
{
    st[u].l = l, st[u].r = r;
    st[u].v.resize(M);
    if (l == r) return;
    int mid = l + r >> 1;
    build(u << 1, l, mid);
    build(u << 1 | 1, mid + 1, r);
}

void modify(int u , int p , int x){
    if(st[u].l == p && st[u].r == p) st[u].v[0] = x;
    else {
        int mid = st[u].l + st[u].r >>  1;
        if(p<=mid) modify(u<<1 , p , x);
        if(p>mid) modify(u<<1|1 , p , x);
        pushup(st[u].v , st[u<<1].v , st[u<<1|1].v);
    }
}

vector<int> query(int u , int l , int r){
    if(st[u].l >= l && st[u].r <= r) return st[u].v;
    else {
        int mid = st[u].l + st[u].r >>  1;
        if (r <= mid) return query(u << 1, l, r);
        if (mid < l) return query(u << 1 | 1, l, r);
        vector<int> ans(8, 0);
        pushup(ans, query(u << 1, l, r), query(u << 1 | 1, l, r));
        return ans;
    }
}
int main()
{
    scanf("%d%d", &n, &m);
    build(1,1,n);
    char op[2];
    int a, b;
    while (m -- )
    {
        scanf("\n%s %d %d", op, &a, &b);
        if (*op == 'C') modify(1, a, b);
        else printf("%d\n", query(1, a, b)[7]);
    }
    return 0;
}
```



# 搜索：

### [bfs：走迷宫+传送门](https://ac.nowcoder.com/acm/contest/12788/H)

```c++
/*
第一行有一个正整数T（1<=T<=200)，代表T组测试用例。
对于每组测试用例：
第一行有两个正整数n和m（2<=n,m<=1000)，表示迷宫的大小。
之后是一个n×m的迷宫，其中：'S'表示起点，'E'表示终点，'#'表示不能走的点，'.'表示可以走的点，'*'表示传送阵。

1
3 3
S.#
*.#
#*E

res = 2
*/
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=1e3+5,M=1e6+7,inf=0x3f3f3f3f,mod=1e9+7;
const int dir[][2]={{0, 1}, {1, 0}, {0, -1}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
#define bug(x) cout << #x << "===" << x << endl
#define Please return
#define AC 0 
#define rep(i, a, n) for (ll i = a; i <= n; i++)
inline int read(){int x = 0, f = 1;char ch = getchar();
    while(ch < '0' || ch > '9'){if (ch == '-')f = -1;ch = getchar();}
    while(ch >= '0' && ch <= '9'){x = (x<<1) + (x<<3) + (ch^48);ch = getchar();}return x * f;}

int t , n , m , sx,sy , ex,ey , xx , xdis , ans = inf , xxx , yyy;
char mp[N][N] , xvis[N][N] , vis[N][N];
struct node{
	int x , y , ans ;
};

bool check(int nx,int ny){
	return nx>=1 && nx<=n && ny>=1 && ny<=m;
}

void bfsx(){
	queue<node> q;
	q.push({ex,ey,0});
	while(!q.empty()){
		node t = q.front();q.pop();
		if(mp[t.x][t.y]=='*') {xdis = t.ans; break;}   //只需扫到最近的传送门 
		rep(i,0,3){
			int nx = t.x+dir[i][0];
			int ny = t.y+dir[i][1];
			if(check(nx,ny) && !xvis[nx][ny] && mp[nx][ny] !='#'){
				xvis[nx][ny] = true;
				q.push({nx,ny,t.ans+1});
			}
		}
	}
}

void bfs(){
	//int f = 0;
	queue<node> q;
	q.push({sx,sy,0});
	while(!q.empty()){
		node t = q.front();q.pop();
		if(mp[t.x][t.y]=='E') {ans = min(ans , t.ans); break;}   
		rep(i,0,3){
			int nx = t.x+dir[i][0];
			int ny = t.y+dir[i][1];
			if(check(nx,ny) && !vis[nx][ny] && mp[nx][ny] !='#'){
				vis[nx][ny] = true;
				q.push({nx,ny,t.ans+1});
				if(mp[nx][ny]=='*'&& xdis!=0) {
					ans = min(ans , t.ans+1 + xdis);   // 如果走传送门，并且传送门的出口能到达E ， 就更新ans 
				} 
				
			}
		}
	}
}

void init(){
	memset(vis,false,sizeof(vis));
	memset(xvis,false,sizeof(xvis));
	xdis = 0 ; xx = 0 , ans = inf;   //一定init哦 
}
int main(){
	t = read();
	rep(temp,1,t){
		init();
		n = read(); m = read();
		rep(i,1,n){
			rep(j,1,m){
				scanf(" %c",&mp[i][j]);
				if(mp[i][j]=='S') sx = i , sy = j;
				if(mp[i][j]=='E') ex = i , ey = j;   // 记录起点和终点 
				if(mp[i][j]=='*') xx=1;   //是否存在传送门 
			}
		}
		
		if(xx) bfsx();    //先扫一次，看看终点到传送门的最短距离 
		//bug(xdis);
		bfs();    //然后从S到E的距离 
		if(ans!=inf) cout<<"Case #"<<temp<<": "<<ans<<endl;
		else cout<<"Case #"<<temp<<": "<<-1<<endl;
		;
		
	}
    Please AC;
}
```

### DFS：树的重心

```c++
/*给定一颗树，树中包含 n 个结点（编号 1∼n）和 n−1 条无向边。
请你找到树的重心，并输出将重心删除后，剩余各个连通块中点数的最大值。
重心定义：重心是指树中的一个结点，如果将这个点删除后，剩余各个连通块中点数的最大值最小，那么这个节点被称为树的重心。
*/
#include<bits/stdc++.h>
using namespace std;
const int N = 1e5+5;
bool vis[N];
int n , ans = N;
int h[N],e[N<<1],ne[N<<1] , cnt;

void add(int a , int b){
    e[cnt] = b; ne[cnt] = h[a] , h[a] = cnt++;
}

int dfs(int u){
    vis[u] = true;
    int sum = 1 , res = 0;
    
    for(int i=h[u]; ~i; i= ne[i]){
        int j = e[i];
        if(!vis[j]){
            int s = dfs(j);   //每个点的子节点都可以回溯返回
            res = max(res , s);
            sum += s;   
        }
    }
    
    res = max(res , n-sum);   //选择u节点为重心，最大的连通子图节点数
    ans = min(ans , res);
    return sum;
}

int main(){
    cin>>n;
    memset(h , -1 , sizeof h);
    for(int i=1;i<n;i++){
        int x ,y ; 
        cin>>x>>y;
        add(x,y) , add(y,x);
    }
    
    dfs(1);
    cout<<ans<<endl;
    return 0;
}
```



### [拐弯次数](https://ac.nowcoder.com/acm/problem/208246)

```c++
/*
给你一个N*N（2≤N≤100）的方格中，‘x’表示障碍，‘.’表示没有障碍（可以走），牛牛可以从一个格子走到他相邻的四个格子，但是不能走出这些格子。问牛牛从A点到B点最少需要转90度的弯几次。

3
. x A
. . .
B x .

//res:2
*/
#include <bits/stdc++.h>
using namespace std;
const int INF = 0x3f3f3f3f;
int dir[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
char mp[105][105];
int vis[105][105][5];
struct node {
    int x, y;
    int step, pre;
};
int ax, ay, bx, by;
void solve() {
    int n; cin >> n;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            cin >> mp[i][j];
            if (mp[i][j] == 'A') ax = i, ay = j;
            if (mp[i][j] == 'B') bx = i, by = j;
        }
    }
    queue<node> q;
    q.push({ax, ay, 0, 4});
    int ans = INF;
    while (q.size()) {
        node now = q.front();
        q.pop();
        vis[now.x][now.y][now.pre] = 1;
        if (now.x == bx && now.y == by) {
            ans = min(ans, now.step);
        }
        for (int i = 0; i < 4; i++) {
            int dx = now.x + dir[i][0];
            int dy = now.y + dir[i][1];
            if (dx >= 1 && dx <= n && dy >= 1 && dy <= n && !vis[dx][dy][i] && mp[dx][dy] != 'x') {
                if (now.pre == 4) q.push({dx, dy, now.step, i});
                else {
                    if (now.pre == 0 || now.pre == 1) {
                        if (i == 2 || i == 3) q.push({dx, dy, now.step+1, i});
                        else q.push({dx, dy, now.step, i});
                    } else {
                        if (i == 0 || i == 1) q.push({dx, dy, now.step+1, i});
                        else q.push({dx, dy, now.step, i});
                    }
                }
            }
        }
    }
    if (ans == INF) cout << -1 << endl;
    else cout << ans << endl;
}

int main() {
    solve();
}


```

### 全排列去重

```c++
#include <bits/stdc++.h>
using namespace std;
class Solution {
public:
    vector<vector<int>> ans;
    vector<int> path;
    vector<bool> st;
    int n ;

    vector<vector<int>> permuteUnique(vector<int>& nums) {
        sort(nums.begin() , nums.end());
        n = nums.size();
        path = vector<int>(n);
        st = vector<bool>(n);
        dfs(nums , 0);
        return ans;
    }

    void dfs(vector<int> nums , int u){
        if(u == n) {
            ans.push_back(path);
            return ;
        }

        for(int i = 0 ; i < n ; i++){
            if(!st[i]){
                if(i && nums[i-1] == nums[i] && !st[i-1]) continue;
                st[i] = true;
                path[u] = nums[i];
                dfs(nums,u+1);
                st[i] = false;
            }
        }
    }
};

/*

给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
示例 1：

输入：nums = [1,1,2]
输出：
[[1,1,2],
 [1,2,1],
 [2,1,1]]
示例 2：

输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

提示：

1 <= nums.length <= 8
-10 <= nums[i] <= 10

*/
```



### N皇后

```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 20;
int n;

int dg[N],udg[N],col[N],row[N];

char g[N][N];

void dfs(int x, int y, int s)
{
    // 处理超出边界的情况
    if (y == n) y = 0, x ++ ;

    if (x == n) { // x==n说明已经枚举完n^2个位置了
        if (s == n) { // s==n说明成功放上去了n个皇后
            for (int i = 0; i < n; i ++ ) puts(g[i]);
            puts("");
        }
        return;
    }

    // 分支1：放皇后
    // x = -y+b   x =y+b
    // b = x+y    b = x-y   出现负数， x-y+n作为映射值
    if (!row[x] && !col[y] && !dg[x + y] && !udg[x - y + n]) {
        g[x][y] = 'Q';
        row[x] = col[y] = dg[x + y] = udg[x - y + n] = true;
        dfs(x, y + 1, s + 1);
        row[x] = col[y] = dg[x + y] = udg[x - y + n] = false;
        g[x][y] = '.';
    }
    // 分支2：不放皇后
    dfs(x, y + 1, s);
}

void dfs1(int u) {
    if (u == n) {
        for (int i = 0; i < n; i ++ ) puts(g[i]);
        puts("");  
        return;
    }
    //对n个位置按行搜索
    for (int i = 0; i < n; i ++ )
        if (!col[i] && !dg[u + i] && !udg[n - u + i]) {
            g[u][i] = 'Q';
            col[i] = dg[u + i] = udg[n - u + i] = true;
            dfs(u + 1);
            col[i] = dg[u + i] = udg[n - u + i] = false; 
            g[u][i] = '.';
        }
}


int main(){
    cin>>n;
    for(int i=0;i<n;i++)
     for(int j=0;j<n;j++) g[i][j]='.';
     
     dfs(0,0,0);
     return 0;
}
```

### 八数码

```c++
#include<bits/stdc++.h>
using namespace std;
const int manx = 362900;
int start[9],goal[9];
int a[manx];
int dir[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
struct node{
    int state[9];     //记录每种状态 
    int dis;          //记录次数 
};

int fac[10];
void init(){ 
     //初始0~9的阶乘   
     fac[0] = fac[1] = 1;
     for(int i=2;i<=9;++i) fac[i] = fac[i-1]*i;
}

//康托展开 
int cantor(int s[],int n){
     int re = 0 ;
     for(int i=0;i<n;i++){
         int t = 0 ;
         for(int j=i+1;j<n;j++){
             if(s[i]>s[j])t++;
         }
         re = re+t*fac[n-i-1];
      }
      re = re+1;
      //判重 
      if(!a[re]){++a[re];return 1;}
      return 0;
}

bool check(int x , int  y){
     return x>=0&&x<3&&y>=0&&y<3;
}

int bfs(){
     cantor(start,9);
     node frist;
     queue<node> q;
     frist.dis=0;
     memcpy(frist.state,start,sizeof(start));
     q.push(frist);
     while(!q.empty()){
          node now = q.front();
          q.pop();
          int z = 0;
          for(;z<9;++z) if(now.state[z]==0) break;
          //得到空格子(0)的坐标 ：一维化二维 
          //例如3*3:0的坐标是(1,0) ,得到x=1，y=0 
          int x = z/3;
          int y = z%3;
          for(int i=0;i<4;i++){
              int nx = x+dir[i][0];
              int ny = y+dir[i][1];
              if(check(nx,ny)){
                  //二维化一维 
                  //0坐标的周围点在一维中的表示 ：xy 
                  int xy = ny+nx*3;  
                  node temp = now;
                  swap(temp.state[z],temp.state[xy]);
                  temp.dis++; 
                  if(memcmp(temp.state,goal,sizeof(goal))==0) return temp.dis;
                  //判重 
                  if(cantor(temp.state,9)) q.push(temp);
               }
           }
       }
       return -1;
}

int main(){
    init();
    for(int i=0;i<9;i++){
        char x;
        cin>>x;
        if(x=='x')start[i] = 0;
        else start[i] = x-'0';
    }
    for(int i=0;i<9;i++) goal[i] = (i+1)%9;
    int ans = bfs();
    cout<<ans<<endl;
} 
```

### [DFS(单词搜索)](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/)

```c++
/*请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。
[["a","b","c","e"],
["s","f","c","s"],
["a","d","e","e"]]
但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。
*/
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        rows = board.size();
        cols = board[0].size();
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if(dfs(board, word, i, j, 0)) return true;
            }
        }
        return false;
    }
private:
    int rows, cols;
    bool dfs(vector<vector<char>>& board, string word, int i, int j, int k) {
        if(i >= rows || i < 0 || j >= cols || j < 0 || board[i][j] != word[k]) return false;
        if(k == word.size() - 1) return true;
        board[i][j] = '\0';
        bool res = dfs(board, word, i + 1, j, k + 1) || dfs(board, word, i - 1, j, k+1)|| dfs(board, word, i, j + 1, k + 1) || dfs(board, word, i , j - 1, k + 1);
        board[i][j] = word[k];
        return res;
    }
};
```

### [tire：单词搜索 II](https://leetcode-cn.com/problems/word-search-ii/)

```c++
#include <bits/stdc++.h>

using namespace std;

class Solution {
public:

    struct Node {      //用一个tire树保存单词列表，剪枝
        int id ;   //单词在列表中的下标
        Node  *son[26];
        Node(){
            for(int i = 0 ; i<26 ; i++) son[i] = NULL;
            id = -1;
        }
    };
    Node *root ;
    int n , m ;
    unordered_set<int> res;   //记录存在的单词的id
    int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};

    void insert(string &word , int idx){    //单词插入tire树
        Node *p = root;
        for(char &x : word){
            int u = x-'a';
            if(!p->son[u]) p->son[u] = new Node();
            p = p->son[u];
        }
        p->id = idx;
    }

    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        root = new Node();
        n = board.size();
        m = board[0].size();
        for(int i = 0 ; i < words.size() ; i++) insert(words[i] , i);

        for(int i = 0 ; i < n ; i++){
            for(int j = 0 ; j < m ; j++){
                int u = board[i][j] - 'a';
                if(root->son[u]) dfs(board , i , j , root->son[u]);
            }
        }

        vector<string> ans;
        for(auto x : res) ans.push_back(words[x]);
        return ans;

    }


    void dfs(vector<vector<char>> &board , int x , int y , Node *p){
        if(p->id != -1) res.insert(p->id);
        char t = board[x][y];
        board[x][y] = '.';
        for(int i = 0 ; i < 4 ; i++){
            int nx = dx[i]+x;
            int ny = dy[i]+y;
            if(nx>=0 && nx<n && ny>=0 && ny<m && board[nx][ny] != '.'){
                int u = board[nx][ny] - 'a';
                if(p->son[u]) dfs(board , nx , ny , p->son[u]);
            }
        }
        board[x][y] = t;
        
    }
};



/*

给定一个 m x n 二维字符网格 board 和一个单词（字符串）列表 words，找出所有同时在二维网格和字典中出现的单词。

单词必须按照字母顺序，通过 相邻的单元格 内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母在一个单词中不允许被重复使用。

示例 1：
输入：board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
输出：["eat","oath"]
示例 2：
输入：board = [["a","b"],["c","d"]], words = ["abcb"]
输出：[]

提示：
m == board.length
n == board[i].length
1 <= m, n <= 12
board[i][j] 是一个小写英文字母
1 <= words.length <= 3 * 104
1 <= words[i].length <= 10
words[i] 由小写英文字母组成
words 中的所有字符串互不相同

*/

```



### DFS(连通个数)

```c++
#include<bits/stdc++.h>
// 连通个数
using namespace std;

int dir[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
//int dir[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
char mp[105][105];
bool vis[105][105];
int n,m;
int ans = 0 ;

bool check(int nx,int ny){
	return nx>=0 && nx<n && ny>=0 && ny<m;
}

void dfs(int x,int y){
	vis[x][y] = true ;
	mp[x][y] = '.';
	for(int i=0;i<8;i++){
		int nx = x+dir[i][0];
		int ny = y+dir[i][1];
		if( check(nx,ny) && !vis[nx][ny] && mp[nx][ny]=='W'){
			dfs(nx,ny);
		}
	}
}

int main(){
	cin>>n>>m;
	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			cin>>mp[i][j];
		}
	}
	
	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			if(mp[i][j]=='W') {
				ans++;
				dfs(i,j);
			}
		}
	}
	
	cout<<ans<<endl;
}

```

### 邻接矩阵连通个数

```c++
#include<bits/stdc++.h>
using namespace std;
int Map[505][505];
int vis[505];
int ans;
int n,m;

void dfs(int x) {
    vis[x] = 1;
    for (int i = 0; i < n; i++) {
        if (!vis[i]&&Map[x][i]) {
            dfs(i);
        }
    }
}

int Count() {
    memset(vis, 0, sizeof(vis));
    int sum = 0;
    for (int i = 0; i < n; i++)
    {
        if (!vis[i]) {
            dfs(i);
            sum++;
        }
    }
    return sum;
}

int main(){

	cin>>n>>m;
	while(m--){
		int u,v;
		cin>>u>>v;
		Map[v][u] = Map[u][v] = 1;
	}
	int Cnt = Count();  
	int k,city;
	cin>>k;
	for(int i=0;i<k;i++){
		cin>>city;
		for(int i=0;i<n;i++){
			Map[city][i] = Map[i][city] = 0;
		} 
		int cnt = Count();
		if(cnt>=Cnt+2) cout<<"Red Alert: City "<<city<<" is lost!"<<endl;
		else cout<<"City "<<city<<" is lost."<<endl;
		Cnt = cnt;
	}
	if(k==n)cout<<"Game Over."<<endl;
}
```

### [DFS(送外卖)](https://ac.nowcoder.com/acm/problem/13224)

```c++
#include <bits/stdc++.h>
using namespace std;
const int N=1e5+5;
int a[N],b[N],st[N],vis[N],flag=0,n;
char ans[N];

bool dfs(int u,int step)
{
    if(u>n||u<1) return false;
    if(u==n){
    	ans[step]='\0';
        return true;
    }
    if(vis[u]){
        st[u]=1;
        return false;
    }
    vis[u]=1;
    // a
    int next = u+a[u];
    if(dfs(next,step+1)){
        ans[step]='a';
        if(st[u]) flag=1;
        return true;
    }
    //b
    next = u+b[u];
    if(dfs(next,step+1)){
        ans[step]='b';
        return true;
    }
    return false;
}
int main(){
    scanf("%d",&n);
    for(int i=1;i<=n;i++)    scanf("%d",&a[i]);
    for(int i=1;i<=n;i++)    scanf("%d",&b[i]);
    if(dfs(1,0)){
        if(flag)    puts("Infinity!");
        else        puts(ans);
    }
    else puts("No solution!");
    return 0;
}
/*n 个小区排成一列，编号为从 0 到 n-1 。一开始，美团外卖员在第0号小区，目标为位于第 n-1 个小区的配送站。
给定两个整数数列 a[0]~a[n-1] 和 b[0]~b[n-1] ，在每个小区 i 里你有两种选择：
1) 选择a：向前 a[i] 个小区。
2) 选择b：向前 b[i] 个小区。
把每步的选择写成一个关于字符 ‘a’ 和 ‘b’ 的字符串。求到达小区n-1的方案中，字典序最小的字符串。如果做出某个选择时，你跳出了这n个小区的范围，则这个选择不合法。
• 当没有合法的选择序列时，输出 “No solution!”。
• 当字典序最小的字符串无限长时，输出 “Infinity!”。
• 否则，输出这个选择字符串。*/
/*
7
5 -3 6 5 -5 -1 6
-6 1 4 -2 0 -2 0
*/
//abbbb

```

### [BFS(数轴)](https://ac.nowcoder.com/acm/problem/107875)

```c++
// 数轴上n点到m点
#include<queue>
#include<iostream>
using namespace std;
const int MAXN = 1e5+10;
int n , m ,ans = 0 ;
bool vis[MAXN];
struct node{
	int k;   //次数 
	int d;   //位置 
};

int bfs(){
	queue<node> q ;
	vis[n] = true;
	node t = {0,n}; 
	q.push(t);
	while(!q.empty()){
		node now = q.front();
		if(now.d==m) return now.k;  // 1 1
		q.pop();
		
		for(int i=0;i<3;i++){
			node temp = now ;
            //约束条件
			if(i==0) temp.d +=-1;
			if(i==1) temp.d +=1;
			if(i==2) temp.d *=2;
			if(temp.d<0 || temp.d > 1e5 || vis[temp.d])continue;
    
			temp.k+=1;
			q.push(temp);
			vis[temp.d] = true;
			if(temp.d==m)return temp.k;
		}
	}
}

int main(){
	cin>>n>>m;
	cout<<bfs()<<endl;
} 
```

### [组合总数](https://leetcode-cn.com/problems/combination-sum-ii/)

```c++
class Solution {
public:
    vector<vector<int>> ans;
    vector<int> path;

    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(),candidates.end());
        dfs(candidates , 0 , target);
        return ans;
    }

    void dfs(vector<int>& candidates , int u , int target){
        if(target == 0){
            ans.push_back(path);
            return ;
        }
        if(u == candidates.size()) return ;

        int k = u+1; // 有多少个
        while(k<candidates.size() && candidates[k] == candidates[u]) k++;
        int cnt = k-u;

        for(int i = 0 ; candidates[u]*i <= target && i<=cnt ; i++){
            dfs(candidates , k , target-candidates[u]*i);
            path.push_back(candidates[u]);
        }

        for(int i = 0 ; candidates[u]*i <= target && i<=cnt ; i++) path.pop_back();
    }
};

/*
给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
candidates 中的每个数字在每个组合中只能使用一次。
说明：
所有数字（包括目标数）都是正整数。
解集不能包含重复的组合。 
示例 1:

输入: candidates = [10,1,2,7,6,1,5], target = 8,
所求解集为:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
*/
```



# 拓扑排序

### [拓扑排序：烦人的依赖](https://ac.nowcoder.com/acm/problem/206078)

```c++
#include<bits/stdc++.h>
using namespace std;
const int maxn = 3e4+4;
vector<int> edge[maxn];
map<string , int > mp; 
string name[maxn];
priority_queue<int,vector<int>,greater<int> > q;
int indeg[maxn];
int t , n , m;
string t1,t2;
int main(){
	cin>>t;
	for(int num = 1 ; num<=t ; num++){
		cin>>n>>m;
		mp.clear(); memset(indeg , 0 ,sizeof(indeg));
		vector<string> ans ;
		for(int  i =1 ; i<=n ;i++) {
			cin>>name[i]; edge[i].clear();
		}
		sort(name+1,name+n+1);
		for(int i=1;i<=n;i++) mp[name[i]] = i;
		for(int i=1;i<=m;i++){
			cin>>t1>>t2;
			++indeg[mp[t2]];
			edge[mp[t1]].push_back(mp[t2]);
		}
		//topsort();
		for(int i=1;i<=n;i++) if(!indeg[i]) q.push(i);
		while(!q.empty()){
			int now = q.top() ; q.pop();
			ans.push_back(name[now]);
			for(int j=0;j<edge[now].size();j++){
				--indeg[edge[now][j]];
				if(!indeg[edge[now][j]]) q.push(edge[now][j]);
			}
		}
		cout<<"Case #"<<num<<":"<<endl;
		if(ans.size()==n){
			for(int i=0;i<ans.size();i++) cout<<ans[i]<<endl;
		}else{
			cout<<"Impossible"<<endl;
		}
	}
} 

/*
t组测试 ， n个软件名 ， m个依赖
cin:
2
4 2
a b c d
a b
b c
3 3
a b c
a b
b c
c a
//////按字典序输入
out:
Case #1:
a
b
c
d
Case #2:
Impossible
*/
```

# 最短路：

### 朴素：Dijkstra

```c++
//1≤n≤500 ,1≤m≤105,求1到n的最短路，

#include<bits/stdc++.h>  //O(n^2)
using namespace std;

int mp[505][505] , dis[505];
int n,m;
bool vis[505];

int dijkstra(){
    
    memset(dis , 0x3f , sizeof dis);  
    dis[1] = 0;  
    
    for(int i=1;i<=n;i++){
        
        int t = -1 ;
        for(int j=1;j<=n;j++)
            if(!vis[j] && (t==-1 || dis[t]>dis[j]))   //找不在集合里门最小的距离
                t = j;    
        
        vis[t] = true;           //加入到集合
        
        for(int j=1;j<=n;j++)    //更新t能到达的点
            dis[j] = min(dis[j] , dis[t]+mp[t][j]);
    }
    
    if(dis[n]==0x3f3f3f3f) return -1;
    else return dis[n];
}

int main(){
    scanf("%d %d",&n,&m);
    memset(mp , 0x3f , sizeof mp);
    for(int i=1;i<=m;i++){
        int a , b , c;
        scanf("%d %d %d",&a,&b,&c);
        mp[a][b] = min(mp[a][b],c);
    }
    int ans = dijkstra();
    cout<<ans<<endl;
    return 0;
}
```

### 堆优化：Dijkstra

```c++
#include<bits/stdc++.h> //O(mlogm)
using namespace std;
const int N = 2e5+5;
typedef pair<int,int> PII;    //距离 ， 编号
int h[N],ne[N<<1],e[N<<1],w[N<<1] , cnt;
int dis[N];
int n,m;
bool vis[N];
void add(int a , int b , int c){
    e[cnt] = b , w[cnt] = c , ne[cnt] = h[a] , h[a] = cnt++;
}
int dijkstra(){
    memset(dis , 0x3f , sizeof dis);  
    dis[1] = 0; 
    priority_queue<PII,vector<PII>,greater<PII> > q;
    q.push({0,1});   //起点
    while(q.size()){
        auto t = q.top();
        q.pop();
        int u = t.second , dist = t.first;
        if(vis[u]) continue;
        vis[u] = true;
        for(int i=h[u] ; ~i ; i = ne[i]){     //松弛
            int j = e[i];
            int d = dist+w[i];
            if(dis[j]> d) dis[j] = d , q.push({d,j});
        }
    }
    if(dis[n]==0x3f3f3f3f) return -1;
    else return dis[n];
}
int main(){
    scanf("%d %d",&n,&m);
    memset(h,-1,sizeof h);
    for(int i=1;i<=m;i++){
        int a , b , c;
        scanf("%d %d %d",&a,&b,&c);
        add(a,b,c);
    }
    int ans = dijkstra();
    cout<<ans<<endl;
    return 0;
}
```

### bellman:边数限制

```c++
/*给定一个 n 个点 m 条边的有向图，图中可能存在重边和自环， 边权可能为负数。
请你求出从 1 号点到 n 号点的最多经过 k 条边的最短距离，
如果无法从 1 号点走到 n 号点，输出 impossible。
*/
#include<bits/stdc++.h> //O(nm)
using namespace std;
const int N = 1e5+5;
int n,m,k;
int dis[N],backup[N];

struct node{
    int x , y , w;
}edge[N];

int bellman_ford(){
    memset(dis,0x3f,sizeof dis);
    dis[1] = 0;
    for(int i=0;i<k;i++){    //k次
        memcpy(backup , dis , sizeof dis);   //防止串联
        for(int j=0;j<m;j++){   //遍历每一条边
            int a = edge[j].x;
            int b = edge[j].y;
            int w = edge[j].w;
            dis[b] = min(dis[b], backup[a] + w);
        }
    }
    
    if(dis[n]>0x3f3f3f3f/2) return 0x3f;  //虽然不能从i到n，但是可以从其他点到n，可以更新dis[n]
    else return dis[n];
    
}
int main(){
    cin>>n>>m>>k;
    for(int i=0;i<m;i++){
        int x , y , z;
        scanf("%d %d %d",&x,&y,&z);
        edge[i] = {x,y,z};
    }
    int ans = bellman_ford();
    if(ans==0x3f) puts("impossible");
    else cout<<ans<<endl;
    return 0;
}
```

### Spfa:最短路+负环

```c++
/*
给定一个 n 个点 m 条边的有向图，图中可能存在重边和自环， 边权可能为负数。
请你求出 1 号点到 n 号点的最短距离，如果无法从 1 号点走到 n 号点，则输出 impossible。
数据保证不存在负权回路。
*/
#include<bits/stdc++.h>//O(m) , 最坏O(nm)
using namespace std;
const int N = 1e5+5;
int e[N],ne[N],w[N],h[N],cnt;
int n,m;
int st[N],dis[N];

void add(int a , int b , int c){
    e[cnt]=b , w[cnt] = c , ne[cnt] = h[a] , h[a] = cnt++;
}

void spfa(){
    
    memset(dis , 0x3f , sizeof dis);
    dis[1] = 0;
    st[1] = true;
    queue<int> q;
    q.push(1);
    //for(int i=1;i<=n;i++) q.push(i); //负环判断
    while(q.size()){
        int t = q.front() ;
        q.pop();
        st[t] = 0;  //从队列中取出来之后该节点st被标记为false,代表之后该节点如果发生更新可再次入队
        for(int i=h[t] ; ~i ; i = ne[i]){
            int j = e[i];
            if(dis[j] > dis[t]+w[i]){
                dis[j] = dis[t]+w[i];
                if(!st[j]){ //当前已经加入队列的结点，无需再次加入队列，即便发生了更新也只用更新数值即可，重复添加降低效率
                    q.push(j);
                   //k[t]++;if(k[t]>=n) return 1; 判断负环
                    st[j] = 1;
                }
            }
        }
    }
}

int main(){
    memset(h,-1,sizeof h);
    scanf("%d %d",&n,&m);
    for(int i=0;i<m;i++){
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        add(a,b,c);
    }
    spfa();
    if(dis[n]==0x3f3f3f3f) puts("impossible");
    else cout<<dis[n]<<endl;
    return 0;
}
```

### Floyd

```c++
/*给定一个 n 个点 m 条边的有向图，图中可能存在重边和自环，边权可能为负数。
再给定 k 个询问，每个询问包含两个整数 x 和 y，表示查询从点 x 到点 y 的最短距离，如果路径不存在，则输出 impossible。*/
#include<bits/stdc++.h> //O(n^3)
using namespace std;
const int N = 205;
int mp[N][N];
int n,m,k;

void init(){
    for(int i=1;i<=n;i++)
        for(int j=1;j<=n;j++)
            if(i!=j) mp[i][j] = 0x3f3f3f3f;
}

void Floyd(){
    for(int k=1;k<=n;k++)
        for(int i=1;i<=n;i++)
            for(int j=1;j<=n;j++)
                mp[i][j] = min(mp[i][j] , mp[i][k]+mp[k][j]);
    //判负环，只要判mp[i][i]是否小于0，即i出去溜达一圈又回到i，变小了
    
}
int main(){
    
    scanf("%d %d %d",&n,&m,&k);
    init();
    for(int i=0;i<m;i++){
        int x , y , z;
        scanf("%d %d %d",&x,&y,&z);
        mp[x][y] = min(mp[x][y] , z);  
    }
    
    Floyd();
    
    while(k--){
        int x , y ;
        scanf("%d %d",&x,&y);
        if(mp[x][y]>0x3f3f3f3f/2) puts("impossible");
        else cout<<mp[x][y]<<endl;
    }
    return 0;
}
```



# 最小生成树：

### prim：朴素

```c++
#include<bits/stdc++.h> //O(n^2)
using namespace std;

const int N = 505 , INF = 0x3f3f3f3f;
int g[N][N];
int dis[N];
int st[N];
int n,m;

int prime(){
    memset(dis , 0x3f , sizeof dis);
    int res = 0;
    
    for(int i=0;i<n;i++){    
        
        int t = -1;
        for(int j=1;j<=n;j++)   //找出集合外距离最小的点
            if(!st[j] && (t==-1 || dis[t] > dis[j])) t = j;
        
        if(i && dis[t] == INF) return INF;    //图不连通
        if(i) res+=dis[t];
        
        for(int j=1;j<=n;j++)   //用t更新其它点到集合的距离
            dis[j] = min(dis[j] , g[t][j]);
        
        st[t] = 1;
    }
    return res;
    
}

int main(){
    scanf("%d %d",&n,&m);
    memset(g,0x3f , sizeof g);
    
    while(m--){
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        g[a][b] = g[b][a] = min(g[a][b] , c);
    }
    
    int ans = prime();
    if(ans==INF) puts("impossible");
    else cout<<ans<<endl;
    
    return 0;
}
```

### [倍增：LCA](https://www.luogu.com.cn/problem/P3379)

```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 5e5+5;
int fa[N][21] , depth[N];  
//fa[i][j] 表示i点第2^j个祖先
//depth[i] 表示i的深度 
int h[N],e[2*N],ne[2*N]; 
//边数切记开2倍空间 
int n,m,root,cnt;
int a,b;
//链式前向星 
void add(int a , int b){
	e[cnt] = b ; ne[cnt] = h[a] ; h[a] = cnt++;
}
//预处理fa和depth，不用dfs是怕爆栈 
void bfs(int root){
	memset(depth , 0x3f , sizeof(depth));
	depth[0] = 0;   // 大有用处，自行理解 
	depth[root] = 1;//根的深度为1 
	int l=0 , r = -1 , q[N];   //模拟队列 
	q[++r] = root;
	while(l<=r){
		int now = q[l++];
		for(int i=h[now];~i;i=ne[i]){
			int j = e[i];
			if(depth[j] > depth[now]+1){
				depth[j] = depth[now]+1;
				q[++r] = j;
				fa[j][0] = now; //j的父节点就是now 
				for(int k=1;k<=20;k++){   
				//此处跳两次，j的2^k那个点
				// 先跳到2^(k-1)，然后再跳2^(k-1) ,递推 
					fa[j][k] = fa[fa[j][k-1]][k-1];
				}
			}
		}
	}
}

int lca(int x , int y){
	//保证x的深度比y深， 
	if(depth[x] < depth[y]) swap(x,y);
	//先x往上跳，跳到y那一层 
	for(int k=20;k>=0;k--){
		int fx = fa[x][k];
		//如果跳了2^k次，深度还是比y的深度大，
		//说明x还是在y的下面，这一步是可行的 
		if(depth[fx] >= depth[y]) x = fx;
	}
	// 如果x和y的lca就是y，直接返回 
	if(x==y) return x;
	for(int k=20;k>=0;k--){
		//x和y同时往上跳， 
		int fx = fa[x][k];
		int fy = fa[y][k];
		if(fx != fy) {
			x = fx;
			y = fy;
		}
	}
	return fa[x][0];
}
int main(){
	scanf("%d %d %d",&n,&m,&root);
	memset(h,-1,sizeof h);
	// n-1 条边
	for(int i=1;i<=n-1;i++){
		scanf("%d %d",&a,&b);
		add(a,b);
		add(b,a);
	}
	bfs(root);
	// m次询问
	while(m--){
		scanf("%d %d",&a,&b);
		printf("%d\n",lca(a,b));
	}
}

```

### [tarjan：LCA](https://www.luogu.com.cn/problem/P3379)

```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 1e5+5 , M = 2*N;
int h[N],e[M],w[M],ne[M];
// dis：存每个点到root的距离
// fa ：并查集
// vis :表示访问的次数 ，0 1 2 
// res : 存储第 i 次的查询的结果 
int dis[N],fa[N],vis[N] , res[N];
int n,m,root=1,cnt;
int a,b,c; 
//first：关系 ， second ：表示id 
typedef pair<int,int> PII;
vector<PII> mp[N];

void add(int a , int b , int c){
	e[cnt] = b, w[cnt]=c , ne[cnt] = h[a] , h[a] = cnt++;
} 

int find(int x){
	return x == fa[x] ? x : fa[x] = find(fa[x]);
} 

// 预处理dis数组,获得每个点到根的距离 
void dfs(int u , int father){
	for(int i = h[u]; ~i ; i=ne[i]){
		int j = e[i];
		if(j==father) continue;
		dis[j] = dis[u]+w[i];
		dfs(j,u);
	} 
}

void tarjan(int u){
	vis[u] = 1;  
	for(int i=h[u]; ~i ; i=ne[i]){
		int j = e[i];
		if(vis[j]==0){   
			tarjan(j);
			fa[j] = u;     // j 缩点到 u 
		}
	}
	
	for(int i=0;i<mp[u].size();i++){
		int y = mp[u][i].first , id =  mp[u][i].second;
		//如果询问x, y, x正在被访问，y已经被访问且已回溯完，则只需要找到y的缩点即可 
		if(vis[y]==2){
			int lca = find(y);  //找到y的缩点 
			res[id] = dis[u]+dis[y] - 2*dis[lca];
		}
	}
	// u已经访问完毕且回溯 
	vis[u] = 2; 
} 

int main(){
	scanf("%d",&n);
	memset(h , -1 , sizeof h);
	for(int i=0;i<n-1;i++){
		scanf("%d %d",&a,&b);
			add(a,b,1);
			add(b,a,1);
	}
	scanf("%d",&m);
	for(int i=0;i<m;i++){
		scanf("%d %d",&a,&b);
		if(a!=b){
			mp[a].push_back({b,i});
			mp[b].push_back({a,i});
		} 
	}
	dfs(root , -1);
	for(int i=1;i<=n;i++) fa[i] = i;   //并查集init 
	tarjan(root);
	for(int i=0;i<m;i++) printf("%d\n",res[i]);
} 
```

### [严格次小生成树](https://www.luogu.com.cn/problem/P4180)

```c++
#include<bits/stdc++.h> 
using namespace std;
typedef long long LL;
const int N = 5e5+5, M = 6e5+5, INF = 0x3f3f3f3f;
int n, m;
struct Edge
{
    int a, b, w;
    bool used; 
    bool operator< (const Edge &t) const{
        return w < t.w;
    }
}edge[M];

int p[N];  //并查集 
int h[N], e[M], w[M], ne[M], idx;
// depth:深度  fa：表示i点第2^j个祖先   d1:保存最大的  d2:保存次大的 
int depth[N], fa[N][17], d1[N][17], d2[N][17];
int q[N];   //队列 

void add(int a, int b, int c)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

int find(int x)
{
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

LL kruskal()    //最小生成树 
{
    for (int i = 1; i <= n; i ++ ) p[i] = i;
    sort(edge, edge + m);
    LL res = 0;
    for (int i = 0; i < m; i ++ )
    {
        int a = find(edge[i].a), b = find(edge[i].b), w = edge[i].w;
        if (a != b)
        {
            p[a] = b;
            res += w;
            //标记此边是最小生成树上的边 
            edge[i].used = true;    
        }
    }
    return res;
}

void build()    //建图 
{
    memset(h, -1, sizeof h);
    for (int i = 0; i < m; i ++ )
        if (edge[i].used)
        {
            int a = edge[i].a, b = edge[i].b, w = edge[i].w;
            add(a, b, w), add(b, a, w);
        }
}


void bfs(int root)   //预处理，fa , depth和 d1 , d2 
{
    memset(depth, 0x3f, sizeof depth);
    depth[0] = 0, depth[root] = 1;
    q[0] = root;
    int hh = 0, tt = 0;
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int j = e[i];
            if (depth[j] > depth[t] + 1)
            {
                depth[j] = depth[t] + 1;  //维护深度 
                q[ ++ tt] = j;    // 入队 
                fa[j][0] = t;    //e[i]的父节点就是 t 
                //因为只有一条边 ，所有最大就是w[i] ，假设次大是负无穷 
                d1[j][0] = w[i], d2[j][0] = -INF;  
                for (int k = 1; k <= 16; k ++ )
                {
                    int anc = fa[j][k - 1];    //e[i]向上跳 2^(k-1)步 
                    fa[j][k] = fa[anc][k - 1]; //再次跳2^(k-1)步，即2^k步 
                    //得到四个数
                    int distance[4] = {d1[j][k - 1], d2[j][k - 1], d1[anc][k - 1], d2[anc][k - 1]};
                    //然后求得里面的最大和次大，
                    d1[j][k] = d2[j][k] = -INF;
                    for (int u = 0; u < 4; u ++ )
                    {
                        int d = distance[u];
                        if (d > d1[j][k]) d2[j][k] = d1[j][k], d1[j][k] = d;
                        // 严格次小生成树，不能等于 
                        else if (d != d1[j][k] && d > d2[j][k]) d2[j][k] = d;
                    }
                }
            }
        }
    }
}

int lca(int a, int b, int w) 
{
    static int distance[N * 2]; 
    int cnt = 0;
    if (depth[a] < depth[b]) swap(a, b);   //先使a,b跳到同一层 
    for (int k = 16; k >= 0; k -- )
        if (depth[fa[a][k]] >= depth[b])   //a向上跳2^j步的深度还是比b大，说明可以跳 
        {
        	//把这一段的最大，次大记录 
            distance[cnt ++ ] = d1[a][k];
            distance[cnt ++ ] = d2[a][k];
            // a更新到祖先(向上2^j) 
            a = fa[a][k];   // 2   
        }
    //此时a和b同时向上跳，跳到最近公共祖先的下面一层 
    if (a != b)
    {
        for (int k = 16; k >= 0; k -- )
        	//如果不相等，说明这步可行，就是没跳到最近公共祖先的上面 
            if (fa[a][k] != fa[b][k])
            {
            	//把每一段的最大和次大记录 
                distance[cnt ++ ] = d1[a][k];
                distance[cnt ++ ] = d2[a][k];
                distance[cnt ++ ] = d1[b][k];
                distance[cnt ++ ] = d2[b][k];
                //同时更新， 
                a = fa[a][k], b = fa[b][k];
            }
        //此时已经跳到最近公共祖先的下面，只有一条边，把a到祖先的边和b到祖先的边记录 
        distance[cnt ++ ] = d1[a][0];
        distance[cnt ++ ] = d1[b][0];
    }
    
    //求这一段里面的最大和次数 
    int dist1 = -INF, dist2 = -INF;  
    for (int i = 0; i < cnt; i ++ )
    {
        int d = distance[i];
        if (d > dist1) dist2 = dist1, dist1 = d;
        else if (d != dist1 && d > dist2) dist2 = d;
    }
    
    //严格次小生成树，返回加进来的边减去最大边 
    if (w != dist1)return w - dist1;
    return w - dist2;
}

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 0; i < m; i ++ )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        edge[i] = {a, b, c};
    }

    LL sum = kruskal();
    build();
    bfs(1);

    LL res = 1e18;
    for (int i = 0; i < m; i ++ )
        if (!edge[i].used)    //每一条不在最小生成树的边去枚举，试着加入 
        {
            int a = edge[i].a, b = edge[i].b, w = edge[i].w;
            res = min(res, sum + lca(a, b, w));
        }
        
    printf("%lld\n", res);
    return 0;
}

```

# 二分图

### 染色法判定二分图

```c++
//给定一个 n 个点 m 条边的无向图，图中可能存在重边和自环。请你判断这个图是否是二分图。O(n+m)
#include<bits/stdc++.h>
using namespace std;
const int N = 1e5+5;
int h[N] ,e[N*2] ,ne[N*2] , cnt;
int col[N];
int n,m;

void add(int a , int b){
    e[cnt] = b , ne[cnt] =  h[a] , h[a] = cnt++;
}

bool dfs(int u , int c){
    col[u] = c;
    for(int i=h[u] ; ~i ; i =ne[i]){
        int j = e[i];
        if(col[j] && col[j] == c) return false;  //如果j已经染过色和u一样，矛盾
        if(!col[j]){
            if(dfs(j,3-c) == false) return false;
        }
    }
    return true;
}

int main(){
    scanf("%d %d",&n,&m);
    memset(h,-1,sizeof h);
    while(m--){
        int a,b;
        scanf("%d %d",&a,&b);
        add(a,b);
        add(b,a);
    }
    
    bool ans = true;
    for(int i=1;i<=n;i++){
        if(!col[i]){      //如果没有染色
            if(!dfs(i , 1)) {   //染色
                ans = false;
                break;
            }
        }
    }
    
    if(ans) puts("Yes");
    else puts("No");
    
    return 0;
}
```

### 匈牙利算法

```c++
/*
给定一个二分图，其中左半部包含 n1 个点（编号 1∼n1），右半部包含 n2 个点（编号 1∼n2），二分图共包含 m 条边。数据保证任意一条边的两个端点都不可能在同一部分中。请你求出二分图的最大匹配数。
*/
#include<iostream>  //O(nm)
#include<cstring>
using namespace std;
const int N = 510 , M = 100010;
int n1,n2,m;
int h[N],ne[M],e[M],idx;
bool st[N];
int match[N]; //match[j]=a,表示女孩j的现有配对男友是a

void add(int a , int b){
    e[idx] = b, ne[idx] = h[a], h[a] = idx++;
}

int find(int x)
{
    //遍历自己喜欢的女孩
    for(int i = h[x] ; i != -1 ;i = ne[i])
    {
        int j = e[i];
        if(!st[j])//如果在这一轮模拟匹配中,这个女孩尚未被预定
        {
            st[j] = true;//那x就预定这个女孩了
            //如果女孩j没有男朋友，或者她原来的男朋友能够预定其它喜欢的女孩。配对成功
            if(!match[j]||find(match[j]))
            {
                match[j] = x;
                return true;
            }

        }
    }
    //自己中意的全部都被预定了。配对失败。
    return false;
}

int main()
{
    memset(h,-1,sizeof h);
    cin>>n1>>n2>>m;
    while(m--){
        int a,b;
        cin>>a>>b;
        add(a,b);
    }

    int res = 0;
    for(int i = 1; i <= n1 ;i ++){  
        memset(st,false,sizeof st);
        if(find(i)) res++;
    }  
    cout<<res<<endl;
}
```



# 背包问题：

### [多重背包问题](https://www.acwing.com/problem/content/5/)

```c++
/*
有 N 种物品和一个容量是 V 的背包。
第 i 种物品最多有 si 件，每件体积是 vi，价值是 wi。
求解将哪些物品装入背包，可使物品体积总和不超过背包容量，且价值总和最大。
0<N≤1000 ,  0<V≤2000   ,0<vi,wi,si≤2000
*/
#include<bits/stdc++.h>
using namespace std;

const int N = 1e5+6;
int dp[N];
int cnt , n , m ;
int v,w,s;
int V[N],W[N];

int main(){
    cin>>n>>m;
    for(int i=1;i<=n;i++){
        cin>>v>>w>>s;
        int k = 1;
        while(k<s){    //二进制分组打包
            cnt++;
            V[cnt] = k*v;
            W[cnt] = k*w;
            s = s-k;
            k = k*2;
        }
        if(s) {
            cnt++;
            V[cnt] = s*v;
            W[cnt] = s*w;
        }
    }
    n = cnt;
    
    for(int i=1;i<=n;i++){   //01背包
        for(int j=m;j>=V[i];j--) dp[j] = max(dp[j] , dp[j-V[i]]+W[i]);
    }
    
    cout<<dp[m]<<endl;
    return 0;
}
```



### [最接近目标值的子序列和](https://leetcode-cn.com/problems/closest-subsequence-sum/)

```c++
/*给你一个整数数组 nums 和一个目标值 goal 。
你需要从 nums 中选出一个子序列，使子序列元素总和最接近 goal 。也就是说，如果子序列元素和为 sum ，你需要 最小化绝对差 abs(sum - goal) 。
返回 abs(sum - goal) 可能的 最小值 。
注意，数组的子序列是通过移除原始数组中的某些元素（可能全部或无）而形成的数组。

示例 1：
输入：nums = [5,-7,3,5], goal = 6
输出：0
解释：选择整个数组作为选出的子序列，元素和为 6 。
子序列和与目标值相等，所以绝对差为 0 。

1 <= nums.length <= 40
-10^7 <= nums[i] <= 10^7
-10^9 <= goal <= 10^9

*/
class Solution {
    //把40个数拆分为20和20，然后求出20个所有子序列和，然后双指针搜索ans
public:
    //枚举所有的和，子集问题
    vector<int> Sum (vector<int> &nums){
        int len = nums.size();
        vector<int>ans(1<<nums.size(),0);

        for(int i=0;i<(1<<len);++i){
            for(int j=0;j<len;j++){
                if(i&(1<<j)) ans[i]+=nums[j];
            }
        }
        sort(ans.begin(),ans.end());
        return ans;
    }

    int minAbsDifference(vector<int>& nums, int goal) {
        int len = nums.size();
        vector<int>left,right;
        for(int i=0;i<len/2;i++) left.push_back(nums[i]);   //前20个
        for(int i=len/2;i<len;i++) right.push_back(nums[i]); //后20个
        //所有的和
        auto Sleft = Sum(left);
        auto Sright = Sum(right);
        int ans=0x3f3f3f3f;

        //双指针找ans
        for(int i=0,j=Sright.size()-1;;){
            int s = Sleft[i]+Sright[j];
            if(abs(s-goal)<ans){
                ans = abs(s-goal);
                if(ans==0) return ans;
            }
            if(s>goal) j--;
            else i++;
            if(j<0||i>=Sleft.size())break;
        }

        return ans;
    }
};
```

### [P722301 背包](https://www.luogu.com.cn/problem/P7223)

```c++
/*有一个容量无穷的背包，往里面放物品，有n个物品，第i个体积为ai，你有一个幸运数字p，若放入的物品体积和为k，你会得到p^k的收益，特别的0^0=1
求所有2^n种放入物品的方案收益和，答案很大，对998244353取模，
1<=n<=10^6 , 0<=p,ai<=998244353
测试样例：
2 2
1 4 

2^0 + 2^1 +2^4 +2^5 = 51

*/
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int mod=998244353;
ll n,m,k,p,s=1;
ll Pow(ll a, ll b){ll ans = 1;while(b > 0){if(b & 1){ans = ans * a % mod;}a = a * a % mod;b >>= 1;}return ans;}
inline int read(){int x = 0, f = 1;char ch = getchar();
    while(ch < '0' || ch > '9'){if (ch == '-')f = -1;ch = getchar();}
    while(ch >= '0' && ch <= '9'){x = (x<<1) + (x<<3) + (ch^48);ch = getchar();}return x * f;}

int main(){
    ios::sync_with_stdio(0);cin.tie(0); 
    n = read();p=read();
	for(int i=1;i<=n;i++){
		m=read();
		//如果我们之前的收益和是sum，对于第i件物品，选它，收益就是 sum*(p^ai),不选就是sum*1
		//所以对于第i件物品，总共收益就是 sum*（p^ai+1) 
		s = s*(Pow(p,m)+1);
		s = s%mod;
	} 
	cout<<s;
	return 0;
}
```

### [P1987 摇钱树](https://www.luogu.com.cn/problem/P1987)

```c++
/*
n棵摇钱树每棵树mi个金币，每天会掉下bi个金币，一天可以砍一棵并获取树上的金币，问k天最大的收益
每组测试数据：
第一行两个整数 n,k。
第二行包含 n 个整数 mi 表示 Cpg 刚看到这n 棵树时每刻树上的金币数。
第三行 n 个整数 bi ，表示每颗摇钱树，每天将会掉落的金币。
1<=k<=n<=10^3,1<=mi<=10^5,1<=bi<=10^3
3 3
10 20 30                --->47
4 5 6

4 3
20 30 40 50            ---->104
2 7 6 5
0 0
*/
#include<bits/stdc++.h>
using namespace std;
const int maxn = 1e3+6;
int dp[maxn][maxn];   //设dp[i][j]为到第 i 棵树时，已选择了 j 棵树的金币的最大值

struct node{
	int mi,bi;
}a[maxn];

bool cmp(struct node a , struct node b){
	return a.bi>b.bi; 
}

int main(){
	int n,k;
	while(cin>>n>>k&&n&&k){
		for(int i=1;i<=n;i++) cin>>a[i].mi;
		for(int i=1;i<=n;i++) cin>>a[i].bi;
		sort(a+1,a+n+1,cmp);      //掉钱多的要先取
		sizeof(dp,0,sizeof(dp));
		
		for(int i=1;i<=n;i++){
			for(int j=k;j>=1;j--){
				int t = a[i].mi - a[i].bi*(j-1);
				t = max(0,t);
				dp[i][j] =  max(dp[i-1][j] , dp[i-1][j-1]+t);
			}
		} 
		int ans = 0;
		for(int i=1;i<=k;i++) ans=max(ans,dp[n][i]);
		cout<<ans<<'\n';
		
	}
	return 0;
}
```



# 字符串：

### [正则表达式](https://leetcode-cn.com/problems/regular-expression-matching/)

```C++
/*
给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。
'.' 匹配任意单个字符
'*' 匹配零个或多个前面的那一个元素
所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。

输入：s = "aa" p = "a*"
输出：true
解释：因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。

输入：s = "aab" p = "c*a*b"
输出：true
解释：因为 '*' 表示零个或多个，这里 'c' 为 0 个, 'a' 被重复一次。因此可以匹配字符串 "aab"。
*/
#include <iostream>
#include <cstring>
#include <algorithm>
#include <unordered_set>

using namespace std;

class Solution {
public:
    bool isMatch(string s, string p) {
        s = " "+s;
        p = " "+p;
        int n = s.size() , m = p .size();
        int dp[n+1][m+1];
        memset(dp, 0, sizeof dp);
        dp[0][0] = 1;
        for(int i = 1 ; i <= n ; i++){
            for(int j = 1 ; j <= m ; j++){
                if(s[i-1] == p[j-1] || p[j-1] == '.') dp[i][j] = dp[i-1][j-1];  // 如果最后一位匹配，只要取决去前面的是否匹配
                else if(p[j-1] == '*'){    //等于*分两种情况
                    if(s[i-1] == p[j-2] || p[j-2] == '.'){
                        dp[i][j]=dp[i][j-1] || dp[i][j-2] || dp[i-1][j];
                        // dp[i][j-1] 匹配1个
                        // dp[i][j-2] 匹配0个
                        // dp[i-1][j] 匹配多个
                    } else {
                        // 如果*号前面那个去当前不匹配，*号只能代表 0 个 
                        //          a  :  ac*
                        dp[i][j] = dp[i][j-2];  
                    }
                }
            }
        }
        return dp[n][m];
    }
};

int main()
{
    Solution solution;
    cout << solution.isMatch("aab","c*a*b") << endl;
    return 0;
}
```



### 马拉夫：最长回文字串

```c++
#include<bits/stdc++.h>
using namespace std;
string str;
char newchar[2402];
int p[2400];
 
void init(){
    int len = str.length();
    int k=0;
    newchar[0] = '$';
    newchar[1]='#';
    int j=2;
    for(int i=0;i<len;i++){
        newchar[j++] = str[i];
        newchar[j++] = '#';
    }
    newchar[j++] = '^';
    newchar[j] = '\0';
}
int manacher(){
    init();
    int len = strlen(newchar);
    int id;
    int mx = 0 ,ans=-1;
    for(int i=1;i<len;i++){
        if(i<mx){
            p[i] = min(p[2*id-i],mx-i);
        }else p[i] = 1;
         
        while(newchar[i-p[i]]==newchar[i+p[i]]) p[i]++;
         
        if(mx<p[i]+i){
            mx = p[i]+i;
            id = i;
        }
        ans = max(ans,p[i]-1);
    }
    return ans;
}
int main(){
	getline(cin,str);
    cout<<manacher()<<endl;
    
}
```

### [DP：最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

```c++
/*
示例 1：
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。

dp:对于一个子串而言，如果它是回文串，并且长度大于 2，那么将它首尾的两个字母去除之后，它仍然是个回文串。对于字符串 "ababa" ，如果已经知道"bab"是回文，那么"ababa"一定是回文，因为首尾字符都是“a"
*/

/*
示例 2：
输入： "aaa"
输出： 6
解释： 6个回文子串: "a", "a", "a", "aa", "aa", "aaa"

*/
class Solution {
public:

    static const int maxn = 1e3+3;
    bool dp[maxn][maxn];           //表示s[i，j]是否是回文
    int ans ;                      //求回文子串的个数
    string longestPalindrome(string s) {
        if(s.size()<2) return s;    //特判
        
        int len = s.size();
        for(int i=0;i<len;i++) dp[i][i] = true,++ans ;    //自己一个字符肯定是回文

        int begin = 0 , maxnLen = 1;
        for(int j=1;j<len;j++){
            for(int i=0;i<j;i++){
                if(s[i]!=s[j]) dp[i][j] = false;   //如果头和尾不一样，不是回文
                else{
                    if(j-i<3) dp[i][j] = true,++ans ; //去掉头尾只有一个字符或者为空，肯定是回文
                    else dp[i][j] = dp[i+1][j-1];  //取决于去掉头尾的子串是不是回文
                }
                
                //如果为真，s[i,j]字串是回文
                if(dp[i][j] && j-i+1 > maxnLen){     //更新一下状态
                    begin = i;
                    maxnLen = j-i+1;
                }
            }
        }
        return s.substr(begin,maxnLen);
    }
};
```

### [DP+回溯：分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning/)

```c++
/*给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。
返回 s 所有可能的分割方案。
示例:
输入: "aab"
输出:
[
  ["aa","b"],
  ["a","a","b"]
]
*/
class Solution {
public:
    static const int maxn = 1e3+3 ;
    bool dp[maxn][maxn];         //dp[i][j] 记录 s[ i-j ] 是否回文 
    vector<vector<string>> res;
    vector<string> path;

    void init(string s){    //预处理
        int len = s.size();
        for(int i=0;i<len;i++) dp[i][i] = true ;
        for(int j=1;j<len;j++){
            for(int i=0;i<j;i++){
                if(s[i]==s[j]&&(dp[i+1][j-1]||j-i<3 )) dp[i][j] = true;
            }
        }
    }

    void dfs(string s , int start){
        if(start == s.length()){
            res.push_back(path);
            return ;
        }

        for(int i=start ; i<s.length() ; i++){
            if(!dp[start][i]) continue ;  //剪枝

            path.push_back(s.substr(start , i-start+1));
            dfs(s,i+1);
            path.pop_back();
        }
    }

    vector<vector<string>> partition(string s) {
        if(!s.size()) return res;
        init(s);
        dfs(s,0 );
        return res;
    }
};
```

### [分割回文串 II](https://leetcode-cn.com/problems/palindrome-partitioning-ii/)

```c++
/*给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。
返回符合要求的最少分割次数。
示例:
输入: "aab"
输出: 1
解释: 进行一次分割就可将 s 分割成 ["aa","b"] 这样两个回文子串。
*/
class Solution {
public:
    int minCut(string s) {
        int len=s.length();
        // 状态定义：dp[i]：前缀子串 s[0:i] （包括索引 i 处的字符）符合要求的最少分割次数
        int dp[len];
        for(int i=0;i<len;i++) dp[i]=i;  ////如果长度n，最大分割次数为n-1

        vector<vector<bool>> ok(len, vector<bool>(len, false));
        for(int j=1;j<len;j++){
            for(int i=0;i<=j;i++){
                if(s[i]==s[j] && (j-i<3 || ok[i+1][j-1])){ 
                    ok[i][j]=true;
                }
            }
        }

        for(int i=0;i<len;i++){
            if(ok[0][i]){
                dp[i]=0;   //s[0][i] 表示子串 0-i是回文，不需要分割
                continue;
            }
            for(int j=0;j<i;j++){
                if(ok[j+1][i]){
                    //dp[j] 表示0-j已经是最优解，
                     //然后 s[j+1][i] 是回文串
                     //所以 dp[i] = min(dp[i] , dp[j]+1)   
                    dp[i]=min(dp[i],dp[j]+1);
                }
            }
        }
        return dp[len-1];
    }
};
```

### [构造最短回文串](https://leetcode-cn.com/problems/shortest-palindrome/)

```c++
#include <bits/stdc++.h>
using namespace std;
class Solution {
public:
    string shortestPalindrome(string s) {
        //KMP
        string t(s.rbegin(), s.rend());
        int n = s.size();
        s = ' ' + s + '#' + t;
        vector<int> ne(n * 2 + 2);
        for (int i = 2, j = 0; i <= n * 2 + 1; i ++ ) {
            while (j && s[i] != s[j + 1]) j = ne[j];
            if (s[i] == s[j + 1]) j ++ ;
            ne[i] = j;
        }
        int len = ne[n * 2 + 1];
        string left = s.substr(1, len), right = s.substr(1 + len, n - len);
        return string(right.rbegin(), right.rend()) + left + right;
    }
};

//hash
/*
一个字符串是回文串，当且仅当该字符串与它的反序相同。因此，我们仍然暴力地枚举s1的结束位置，并计算s1反序s1的哈希值。如果这两个哈希值相等，说明我们找到了一个 s的前缀回文串。
*/
class Solution {
public:
    string shortestPalindrome(string s) {
        int n = s.size();
        int base = 131, mod = 1000000007;
        int left = 0, right = 0, mul = 1;
        int best = -1;
        for (int i = 0; i < n; ++i) {
            left = ((long long)left * base + s[i]) % mod;
            right = (right + (long long)mul * s[i]) % mod;
            if (left == right) {
                best = i;
            }
            mul = (long long)mul * base % mod;
        }
        string add = (best == n - 1 ? "" : s.substr(best + 1, n));
        reverse(add.begin(), add.end());
        return add + s;
    }
};


/*
给定一个字符串 s，你可以通过在字符串前面添加字符将其转换为回文串。找到并返回可以用这种方式转换的最短回文串。
示例 1：

输入：s = "aacecaaa"
输出："aaacecaaa"
示例 2：
输入：s = "abcd"
输出："dcbabcd"

提示：
0 <= s.length <= 5 * 104
s 仅由小写英文字母组成
*/

```



### [字符串字串](https://ac.nowcoder.com/acm/contest/9700/B)

```c++
/*
给定一个仅由大写字母和小写字母组成的字符串。
一个字符串是“牛”的，当且仅当其有一个子串为“NowCoder”（区分大小写）。
问给定字符串有多少个子串是“牛”的。
NowCoderNowCode   ----  8
*/
#include <bits/stdc++.h>
char p[] = "NowCoder";
using ll = long long;
char s[1000005];
int n;
ll ans;

int main() {
	scanf("%s",s+1);
	n = std::strlen(s+1);
	int last = 0;
	for (int i = 1; i <= n; ++ i) {
		if (i >= 8) {
			int flag = 1;
			for (int j = 1; j <= 8; ++ j) 
				if (s[i - j + 1] != p[8 - j]) { flag = 0; break; }
			if (flag) last = i - 8 + 1;
		} ans += last;
		
	} printf("%lld",ans);
	return 0;
}
```

### 尺取法

```c++
#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    string minWindow(string s, string t) {
        unordered_map<char,int> hs , ht;
        for(auto ch : t) ht[ch]++;

        string ans;
        int cnt = 0 ;  // 几个数
        //双指针
        for(int i = 0 , j = 0 ; i < s.size() ; i++){
            hs[s[i]]++;
            if(hs[s[i]] <= ht[s[i]]) cnt++;

            while(hs[s[j]] > ht[s[j]]) hs[s[j++]] -- ;

            if(cnt == t.size()){
                if(ans.empty() || i-j+1 < ans.size()) 
                ans = s.substr(j,i-j+1);
            }
        }
        return ans;
    }
};

/*

给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

注意：如果 s 中存在这样的子串，我们保证它是唯一的答案。

示例 1：

输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
示例 2：

输入：s = "a", t = "a"
输出："a"
 
提示：
1 <= s.length, t.length <= 105
s 和 t 由英文字母组成
*/
```

### [K-beautiful Strings](https://vjudge.net/contest/426462#problem/C)

```c++
/*
找到一个长度与原串一样，比原串大字典序最小的字符串，并且字符串每个字符的个数能 %k = 0
*/
#include<bits/stdc++.h>
using namespace std;
int n ,k ;
string str;
int get(int cnt){
	return (k-(cnt%k))%k;
}
int cnt[30];
void solve(){
	if(n%k){puts("-1");return ;}
	int sum = 0;
	memset(cnt,0,sizeof(cnt));
	for(int i=0;i<n;i++)cnt[str[i]-'a']++;
	for(int i=0;i<26;i++) sum+=get(cnt[i]);
	if(!sum) {cout<<str<<endl;return ;}
	
	for(int i=n-1;i>=0;i--){
		sum = sum - get(cnt[str[i]-'a']) + get(--cnt[str[i]-'a']);
		for(int j=str[i]-'a'+1 ; j<26 ; j++){
			int tsum = sum;
			sum = sum - get(cnt[j]) + get(++cnt[j]);
			if(sum+i < n) {
				for(int x=0;x<i;x++) cout<<str[x];
				cout<<char('a'+j);
				for(int a=0;a<n-sum-i-1;a++) cout<<'a';
				
				for(int w=0;w<26;w++){
					int f = get(cnt[w]);
					while(f--) cout<<char('a'+w);
				}
				cout<<endl;
				return ;
			} 
			sum = tsum ; cnt[j]--;
		}
	}
}
int main(){
	int t;
	cin>>t;
	while(t--){
		cin>>n>>k;
		cin>>str;
		solve();
	}
}
```

# GCD

### 快速GCD

```c++
#include <iostream>
#include <algorithm>
using namespace std;
typedef long long ll;
ll qGCD(ll a, ll b)
{
	if(a == 0) return b;
	if(b == 0) return a;
	if(!(a & 1) && !(b & 1)) // a % 2 == 0 && b % 2 == 0;
		return qGCD(a >> 1, b >> 1) << 1;
	else if(!(b & 1))
		return qGCD(a, b >> 1);
	else if(!(a & 1))
		return qGCD(a >> 1, b);
	else
		return qGCD(abs(a - b), min(a, b));
}

```

### [GCD of an Array](https://vjudge.net/problem/CodeForces-1493D)

```c++
/*
n个数，m次操作输入i，x
对第i个数乘以x，然后打印数列的gcd mod 1e9+7
1<= ai,n,m,i,x <= 2*1e5
*/
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll mod=1e9+7;
ll qpow(ll a,ll n){ll r=1;while(n){if(n&1)r=(r*a)%mod;n>>=1;a=(a*a)%mod;}return r;}

multiset<int> st[200050]; //记录每个因子在每个位置出现的次数，未出现则不需要插入
int pre[200050]; //附加在st上，用于记录上一状态某质因子出现的次数，与st首元素的差值可用于计算答案
map<int,int> mp[200050]; //记录每个位置每个质因子目前的幂次

vector<int> primvec; //素数
bool vis[200050];

void init() //素数筛，素数存入primvec内
{
    vis[0]=vis[1]=true;
    for(int i=2;i<=1000;i++)
    {
        if(!vis[i])
        {
            primvec.push_back(i);
            for(int j=i*i;j<=200000;j+=i)
                vis[j]=true;
        }
    }
    for(int i=1001;i<=200000;i++)
        if(!vis[i])
            primvec.push_back(i);
}

void solve()
{
    int n,q;
    cin>>n>>q;
    ll ans=1;
    
    for(int i=1;i<=n;i++)
    {
        int d;
        cin>>d;
        for(int j:primvec) //对于每个输入的数字，进行一次质因子分解
        {
            if(d==1) //直接跳出循环节省时间
                break;
            if(!vis[d]) //如果可以直接判断为素数，尽量直接跳出循环
            {
                mp[i][d]=1;
                st[d].insert(1);
                break;
            }
            int t=0;
            while(d%j==0) //分解质因子j，记录次数
            {
                d/=j;
                t++;
            }
            if(t)
            {
                mp[i][j]=t; //第i个位置的质因子j的幂次为t
                st[j].insert(t); //记录质因子j在每个位置出现的次数
            }
        }
    }
    
    for(int j:primvec)
    {
        if(st[j].size()==n) //如果质因子j在每个位置都出现至少一次
        {
            int tmp=*st[j].begin(); //此时最少的出现次数
            if(tmp!=pre[j]) //与pre做差，得到答案增幅
            {
                ans=ans*qpow(j,tmp-pre[j])%mod;
                pre[j]=tmp;
            }
        }
    }
    
    while(q--)
    {
        int p,d;
        cin>>p>>d;
        for(int j:primvec) //对d进行质因子分解
        {
            if(d==1) //此处优化同上
                break;
            if(!vis[d])
            {
                int &v=mp[p][d]; //直接引用以优化常数
                if(v) //如果mp[p][d]不为0，说明质因子d已经有幂次，需要将其先从st[d]中删去
                    st[d].erase(st[d].lower_bound(v));
                v++;
                st[d].insert(v);
                
                if(st[d].size()==n) //如果质因子j在每个位置都出现至少一次，下同
                {
                    int tmp=*st[d].begin();
                    if(tmp!=pre[d])
                    {
                        ans=ans*qpow(d,tmp-pre[d])%mod;
                        pre[d]=tmp;
                    }
                }
                break;
            }
            int t=0;
            while(d%j==0)
            {
                d/=j;
                t++;
            }
            if(t) //下同
            {
                int &v=mp[p][j];
                if(v)
                    st[j].erase(st[j].lower_bound(v));
                v+=t; //应增加t次
                st[j].insert(v);
                
                if(st[j].size()==n)
                {
                    int tmp=*st[j].begin();
                    if(tmp!=pre[j])
                    {
                        ans=ans*qpow(j,tmp-pre[j])%mod;
                        pre[j]=tmp;
                    }
                }
            }
        }
        cout<<ans<<'\n';
    }
}
int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);cout.tie(0);
    init();
    solve();
    return 0;
}
```

# 前后缀分解

### [买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<int> f(n+2);  // f[i] 表示前i天一次买卖最大收益

        // 前后缀分解
        for(int i = 1 , mi = INT_MAX ; i <= n ; i++){
            f[i] = max(f[i-1] , prices[i-1] - mi);
            mi = min(mi , prices[i-1]);
        }

        //以i为分界点，i~i为取max ， i ~ n 取一个max
        int res = 0;
        for(int i = n , mx = 0 ; i ; i--){
            res = max(res , max(0,mx - prices[i-1]) + f[i-1]);
            mx = max(mx , prices[i-1]);
        }

        return res;
    }
};
/*
给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
示例 1:

输入：prices = [3,3,5,0,0,3,1,4]
输出：6
解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
     随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。
1 <= prices.length <= 10^5
0 <= prices[i] <= 10^5
*/
```



# 无家可归：

### [逃选方案问题](https://ac.nowcoder.com/acm/contest/10323/B)

```c++
/*
自助餐厅里有5个盘子，里面装的都是面包。
第1个盘子里有无限个面包；
第2个盘子里只有1个面包；
第3个盘子里只有4个面包；
第4个盘子里也有无限个面包，但必须两个两个地拿；
第5个盘子里也有无限个面包，但必须5个5个地拿；
给定正整数n，求有多少种正好拿出n个面包的方案。
方案a和方案b不同，当且仅当方案a存在从某个盘子里拿出面包的数量与方案b中对应盘子拿出的数量不同。
*/
long long wwork(int n) {
        // write code here
        return 1LL*(n+2)*(n+1)/2;
}
```



### [将数组分成三个子数组的方案数](https://leetcode-cn.com/problems/ways-to-split-array-into-three-subarrays/)

```c++
/*
我们称一个分割整数数组的方案是 好的 ，当它满足：
数组被分成三个 非空 连续子数组，从左至右分别命名为 left ， mid ， right 。
left 中元素和小于等于 mid 中元素和，mid 中元素和小于等于 right 中元素和。
给你一个 非负 整数数组 nums ，请你返回 好的 分割 nums 方案数目。由于答案可能会很大，请你将结果对 109 + 7 取余后返回。

输入：nums = [1,2,2,2,5,0]
输出：3
解释：nums 总共有 3 种好的分割方案：
[1] [2] [2,2,5,0]
[1] [2,2] [2,5,0]
[1,2] [2,2] [5,0]

思路：前缀和 + 二分
根据题意，需要用到某段区间内的元素和，所以提前维护一个前缀和数组 v，v[i] 记录给定数组中 [0, i - 1 区间的元素和（其中 i = 1, 2, ...）。
遍历 n 个位置，将位置 i 作为 left 和 mid 子数组的分界线，二分查找 mid和 right 的分界线范围。
假设分界线范围为 [x, y]，则 x 位置需满足的条件是 v[x] - v[i] >=v[i]，此时才能满足 left < mid，同理 y 位置需满足的条件是 v[n] - v[y] >=v[y] - v[i]，这样才能满足 mid < right。求出 x, y 后，这段区间内的任一位置都能作为 mid 和 right 的分界线。
*/

class Solution {
public:
    int waysToSplit(vector<int>& a) {
        int n = a.size();
        vector<int> v(n + 1, 0);
        // 初始化前缀和数组
        for(int i = 1; i <= n; i++) {
            v[i] = v[i - 1] + a[i - 1];
        }
        // 返回值 ret
        long long ret = 0;
        const int M = 1e9 + 7;
        
        for(int i = 1; i < n; i++) {
            // 特判
            if(v[i] * 3 > v[n]) break;
            
            // 第一次二分找右边界
            int l = i + 1, r = n - 1;
            while(l <= r) {
                int mid = (l + r) / 2;
                if(v[n] - v[mid] < v[mid] - v[i]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            }
            // 第二次二分找左边界
            int ll = i + 1, rr = n - 1;
            while(ll <= rr) {
                int mid = (ll + rr) / 2;
                if(v[mid] - v[i] < v[i]) {
                    ll = mid + 1;
                } else {
                    rr = mid - 1;
                }
            }
            ret += l - ll;
        }
        return ret % M;
    }   
};
```

### [格雷编码](https://leetcode-cn.com/problems/gray-code/)

```c++
class Solution {
public:
    vector<int> grayCode(int n) {
        vector<int> res(1,0);
        while(n--){
            //镜像翻转
            for(int i = res.size()-1 ; i>=0 ;i--){
                res[i] *= 2;   //前一半补0
                res.push_back(res[i] + 1 ); // 后一半补1
            }
        }
        return res;
    }
};
/*
格雷编码是一个二进制数字系统，在该系统中，两个连续的数值仅有一个位数的差异。
给定一个代表编码总位数的非负整数 n，打印其格雷编码序列。即使有多个不同答案，你也只需要返回其中一种。
格雷编码序列必须以 0 开头。

示例 1:

输入: 2
输出: [0,1,3,2]
解释:
00 - 0
01 - 1
11 - 3
10 - 2

对于给定的 n，其格雷编码序列并不唯一。
例如，[0,2,3,1] 也是一个有效的格雷编码序列。

00 - 0
10 - 2
11 - 3
01 - 1
示例 2:

输入: 0
输出: [0]
解释: 我们定义格雷编码序列必须以 0 开头。
     给定编码总位数为 n 的格雷编码序列，其长度为 2n。当 n = 0 时，长度为 20 = 1。
     因此，当 n = 0 时，其格雷编码序列为 [0]。
*/
```

