#include <vector>
#include <string>
#include <iostream>
using namespace std;

struct Person
{
    string name;
    int age;
    //初始构造函数
    Person(string p_name="null", int p_age=-1): name(std::move(p_name)), age(p_age)
    {
         cout << "I have been constructed" <<endl;
    }
     //拷贝构造函数
     Person(const Person& other): name(std::move(other.name)), age(other.age)
    {
         cout << "I have been copy constructed" <<endl;
    }
     //转移构造函数
     Person(Person&& other): name(std::move(other.name)), age(other.age)
    {
         cout << "I have been moved"<<endl;
    }
};

int main()
{
    vector<Person> e;
    Person a("Jack", 21);
    cout << "emplace_back:" <<endl;
    e.emplace_back(a); //不用构造类对象
    a.name = "Jack1";
    cout << "a.name: " << a.name << endl;
    cout << "e[0].name: " << e[0].name << endl;

    vector<Person> p(5);//首先Person拥有默认构造函数
    cout << "Vec p.size() = " << p.size() << "\t" << "Vec p.capacity = " << p.capacity() << endl;
    cout << "push_back(Person(\"Mike\",36)) :"<<endl;
    p.push_back(Person("Mike",36));//这里触发了一次扩容机制,导致一次数据的整体的迁移,同时预留原来一倍的空余容量4->8
    cout << "Vec p.size() = " << p.size() << "\t" << "Vec p.capacity = " << p.capacity() << endl;
    cout << "push_back(a) :"<<endl;
    p.push_back(a);
    cout << "Vec p.size() = " << p.size() << "\t" << "Vec p.capacity = " << p.capacity() << endl;
    cout << "push_back(a) :"<<endl;
    p.push_back(a);
    cout << "Vec p.size() = " << p.size() << "\t" << "Vec p.capacity = " << p.capacity() << endl;
    cout << "push_back(a) :"<<endl;
    p.push_back(a);
    cout << "Vec p.size() = " << p.size() << "\t" << "Vec p.capacity = " << p.capacity() << endl;
    cout << "push_back(a) :"<<endl;
    p.push_back(a);
    cout << "Vec p.size() = " << p.size() << "\t" << "Vec p.capacity = " << p.capacity() << endl;
    p.push_back(a);//在这次操作之后也会触发一次扩容操作
    cout << "Vec p.size() = " << p.size() << "\t" << "Vec p.capacity = " << p.capacity() << endl;
    
    return 0;
}
// I have been constructed
// emplace_back:
// I have been copy constructed
// a.name: Jack1
// e[0].name: Jack
// I have been constructed
// I have been constructed
// I have been constructed
// I have been constructed
// push_back(Person("Mike",36)) :
// I have been constructed
// I have been moved
// I have been copy constructed
// I have been copy constructed
// I have been copy constructed
// I have been copy constructed
// push_back(a) :
// I have been copy constructed
// push_back(a) :
// I have been copy constructed
// push_back(a) :
// I have been copy constructed
// push_back(a) :
// I have been copy constructed
// I have been copy constructed
// I have been copy constructed
// I have been copy constructed
// I have been copy constructed
// I have been copy constructed
// I have been copy constructed
// I have been copy constructed
// I have been copy constructed
