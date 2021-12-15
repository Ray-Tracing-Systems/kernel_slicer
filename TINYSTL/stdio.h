#ifndef _STDIO_H
#define _STDIO_H

#include "FILE.h"

/* Standard streams.  */
extern FILE *stdin;		/* Standard input stream.  */
extern FILE *stdout;		/* Standard output stream.  */
extern FILE *stderr;		/* Standard error output stream.  */
/* C89/C99 say they're macros.  Make them happy.  */
#define stdin stdin
#define stdout stdout
#define stderr stderr

/* Remove file FILENAME.  */
extern int remove (const char *__filename);
/* Rename file OLD to NEW.  */
extern int rename (const char *__old, const char *__new);

#endif
