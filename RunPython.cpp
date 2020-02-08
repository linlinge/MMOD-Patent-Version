#include "RunPython.h"

void LoadPython(vector<double> dat)
{
    Py_Initialize();//使用python之前，要调用Py_Initialize();这个函数进行初始化
    if (!Py_IsInitialized())
    {
        printf("Init Failed! ");
        // return 0;
    }
    
    PyRun_SimpleString("import sys");
    //这一步很重要，修改Python路径
    PyRun_SimpleString("sys.path.append('/home/llg/workspace_cmake/P03-Improved-LOOP/scripts/')");
    
    PyObject * pModule = NULL;  //声明变量
    PyObject * pFunc = NULL;    // 声明变量

    pModule = PyImport_ImportModule("myfun");   //这里是要调用的文件名hello.py
    if (pModule == NULL){
        cout<<"Module Load Failed!"<< endl;
    }

    pFunc = PyObject_GetAttrString(pModule, "wokaka");//这里是要调用的函数名
    if(pFunc==NULL){
        cout<<"Function Load Failed!"<<endl;
    }
    
    PyObject* args = PyTuple_New(1);
    PyObject* pyListX = PyList_New(dat.size());
    for(int i=0;i<dat.size();i++){
        PyList_SetItem(pyListX, i, PyFloat_FromDouble(dat[i]));
    }
    
    PyTuple_SetItem(args, 0, pyListX);
    PyObject* pRet = PyObject_CallObject(pFunc, args);//调用函数
    int retLen=PyList_Size(pRet);
    cout<<retLen<<endl;
   

    Py_Finalize();
}