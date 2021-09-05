#include <iostream>
#include <fstream>

void test_class_cpu();
void test_class_gpu();

template<class T>
class range 
{
 public:

   class iterator 
   {
      friend class range;
    public:
      T operator *() const { return i_; }
      const iterator &operator ++() { ++i_; return *this; }
      iterator operator ++(T) { iterator copy(*this); ++i_; return copy; }

      bool operator ==(const iterator &other) const { return i_ == other.i_; }
      bool operator !=(const iterator &other) const { return i_ != other.i_; }

    protected:
      iterator(T start) : i_ (start) { }

    private:
      T i_;
   };

   iterator begin() const { return begin_; }
   iterator end() const { return end_; }
   range(T  begin, T end) : begin_(begin), end_(end) {}

private:
   iterator begin_;
   iterator end_;
};

int main(int argc, const char** argv)
{
  //for (auto i : range(1,10)) 
  //  std::cout << i << " ";
  //std::cout << std::endl;

  test_class_cpu();
  //test_class_gpu();
  return 0;
}
