#ifndef __STDARG_H
#define __STDARG_H

#ifndef _VA_LIST
typedef int va_list[1];
#define _VA_LIST
#endif

#define va_start(ap, param) (void)
#define va_end(ap)          (void)
#define va_arg(ap, type)    (void)

#endif /* __STDARG_H */
