#include <bits/stdc++.h>
#define all(x) x.begin(),x.end()
#define msg(str,str2) cout << str << str2<< endl
using namespace std;

using ll = long long;
using ld = long double;
using uint = unsigned int;
using ull = unsigned long long;
template<typename T>
using pair2 = pair<T, T>;
using pii = pair<int, int>;
using pli = pair<ll, int>;
using pll = pair<ll, ll>;

#define pb push_back
#define mp make_pair

int gcd(int a,int b){
	if(a%b==0) return b;
	else return gcd(b,a%b);
}

clock_t startTime;
double getCurrentTime() {
	return (double)(clock() - startTime) / CLOCKS_PER_SEC;
}

vector<complex<double>> f = {0, 0.707, 1, 0.707, 0, -0.707, -1, -0.707};
vector<complex<double>> F;
int M = 8;
const double PI = 3.14159265358979323846;
complex<double> j = 0.0 + 1.0i; // Unidad imaginaria



complex<double> fourier(double u) {
	complex<double> sum = 0;
	for (double x = 0; x < M; x++) {
		double part = 2 * PI * u * x / M;
		// double c_part = part;
		complex<double> cd = -j * complex<double>(part);
		// complex<double> fx = f[x];
		sum += ( complex<double>(f[x]) * complex<double>(exp(cd)));
	}
	return sum;
}


double ang_45 = PI/4;


complex<double> fourier_senos_cosenos(double u) {
	complex<double> sum = 0;
	for (double x = 0; x < M; x++) {
		double angle = 2 * PI * u * x / M;
		complex<double> tri = complex<double>(cos(angle)) -j*complex<double>(sin(angle));
		// complex<double> c_tri = tri;
		sum += (f[x]*tri);
	}
	return sum;
}


void solve(){
	F.resize(M);
	for(int i = 0; i < M; i++){
		// F[i] = fourier(i);
		// cout << F[i] << endl;
		F[i] = fourier_senos_cosenos(i);
	}
	// print fourier transform
	for(auto e: F){
		cout << e << endl;
	}

}


int main(){
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	#ifdef DEBUG
		freopen("input.txt","r",stdin);
		freopen("output.txt","w",stdout);
	#endif
	solve();
	return 0;
}