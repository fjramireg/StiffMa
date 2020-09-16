/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#include "params.h"

static char **help_strings = NULL;
static char **param_strings = NULL;
static char **default_strings = NULL;
static char **type_strings = NULL;
static int n_params = 0;

static int add_param(const char *param, const char* help_str, const char *def, const char *type)
{
  size_t slen;

  if(!help_strings){
    mcalloc(help_strings, sizeof(char*));
    mcalloc(param_strings, sizeof(char*));
    mcalloc(default_strings, sizeof(char*));
    mcalloc(type_strings, sizeof(char*));
  } else {
    mrealloc(help_strings, (n_params+1)*sizeof(char*), sizeof(char*));
    mrealloc(param_strings, (n_params+1)*sizeof(char*), sizeof(char*));
    mrealloc(default_strings, (n_params+1)*sizeof(char*), sizeof(char*));
    mrealloc(type_strings, (n_params+1)*sizeof(char*), sizeof(char*));
  }

  /*    store help string */
  slen = strlen(help_str);
  mcalloc(help_strings[n_params], slen+1);
  strncpy(help_strings[n_params], help_str, slen);

  /*    store parameter name */
  slen = strlen(param);
  mcalloc(param_strings[n_params], slen+1);
  strncpy(param_strings[n_params], param, slen);

  /*    store default values */
  slen = strlen(def);
  mcalloc(default_strings[n_params], slen+1);
  strncpy(default_strings[n_params], def, slen);

  /*    store value type */
  slen = strlen(type);
  mcalloc(type_strings[n_params], slen+1);
  strncpy(type_strings[n_params], type, slen);

  n_params++;

  return 0;
}


void params_free()
{
  int param;
  size_t slen;

  for(param=0; param<n_params; param++){
    slen = strlen(help_strings[param]);
    mfree(help_strings[param], slen);

    slen = strlen(param_strings[param]);
    mfree(param_strings[param], slen);

    slen = strlen(default_strings[param]);
    mfree(default_strings[param], slen);

    slen = strlen(type_strings[param]);
    mfree(type_strings[param], slen);
  }
  mfree(help_strings, n_params*sizeof(char*));
  mfree(param_strings, n_params*sizeof(char*));
  mfree(default_strings, n_params*sizeof(char*));
  mfree(type_strings, n_params*sizeof(char*));
}


void param_print_help(int argc, char **argv)
{
  int i=0;

  while(i<argc){
    if(!strcmp(argv[i], "-help")){
      int param;
      printf("\nusage: %s <parameters>\n", argv[0]);
      printf("------------------------------------------------------------------------------------\n");
      printf("parameters:\tvalue\t\tdefault value\tdescription\n");
      printf("------------------------------------------------------------------------------------\n");
      for(param=0; param<n_params; param++){
	printf("%-15s%10s%10s\t\t%s\n", param_strings[param], type_strings[param], default_strings[param], help_strings[param]);
      }
      exit(0);
    }
    i++;
  }
}


int param_get(int argc, char **argv, const char *param, const char *help_str)
{
  int i=0;
  add_param(param, help_str ,"", "none");

  while(i<argc){
    if(!strcmp(argv[i], param)){
      return 1;
    }
    i++;
  }
  return 0;
}


int param_get_int(int argc, char **argv, const char *param, const char *help_str, int def)
{
  int i=0;

  char buff[256]; 
  SNPRINTF(buff, 255, "%d", def);

  add_param(param, help_str, buff, "int");

  while(i<argc){
    if(!strcmp(argv[i], param)){
      i++;
      return atoi(argv[i]);
    }
    i++;
  }
  return def;
}


double param_get_double(int argc, char **argv, const char *param, const char *help_str, double def)
{
  int i=0;

  char buff[256]; 
  SNPRINTF(buff, 255, "%.3lf", def);

  add_param(param, help_str, buff, "double");

  while(i<argc){
    if(!strcmp(argv[i], param)){
      i++;
      return strtod(argv[i], NULL);
    }
    i++;
  }
  return def;
}


const char *param_get_string(int argc, char **argv, const char *param, const char *help_str)
{
  int i=0;

  add_param(param, help_str, "", "string");

  while(i<argc){
    if(!strcmp(argv[i], param)){
      i++;
      return argv[i];
    }
    i++;
  }
  return NULL;
}


