#include <cstdio>
#include <iostream>

#include <GL/glew.h>
#include <SDL.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "kernel.h"


using namespace std;


#define W 1920
#define H 1080

void sdldie(const char *msg)
{
  printf("%s: %s\n", msg, SDL_GetError());
  SDL_Quit();
  exit(1);
}


void checkSDLError(int line = -1)
{
#ifndef NDEBUG
  const char *error = SDL_GetError();
  if(*error != '\0')
  {
    printf("SDL Error: %s\n", error);
    if(line != -1)
      printf(" + line: %i\n", line);
    SDL_ClearError();
  }
#endif
}


#define CHECK_OPENGL_ERROR \
{ GLenum error; \
  while ((error = glGetError()) != GL_NO_ERROR) { \
  printf("OpenGL ERROR: %s\nCHECK POINT: %s (line %d)\n", \
  gluErrorString(error), __FILE__, __LINE__); \
  } \
}

int main(int argc, char *argv[])
{
  cudaGLSetGLDevice(0);

  SDL_Window *mainwindow; // Our window handle
  SDL_GLContext maincontext; // Our opengl context handle

  if(SDL_Init(SDL_INIT_VIDEO)< 0){
    sdldie("Unable to initialize SDL"); // Or die on error
  }

  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  //SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

  mainwindow = SDL_CreateWindow("Snow", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
    W, H, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN  | SDL_WINDOW_FULLSCREEN); //
  if(!mainwindow){
    sdldie("Unable to create window");
  }

  checkSDLError(__LINE__);

  // Create our opengl context and attach it to our window
  maincontext = SDL_GL_CreateContext(mainwindow);
  checkSDLError(__LINE__);

  // loads OpenGL extensions to support buffers.
  // glewInit()crashes if called before SDL_GL_CreateContext().
  glewExperimental=GL_TRUE;
  GLenum err=glewInit();
  if(err!=GLEW_OK)
  {
    cout << "glewInit failed, aborting." << endl;
  }

  // This makes our buffer swap syncronized with the monitor's vertical refresh
  SDL_GL_SetSwapInterval(1);

  // Set up for 2D drawing.
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity(); 
  glOrtho(0, W, H, 0, 0, 1); 
  glDisable(GL_DEPTH_TEST);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity(); 
  // Displacement trick for exact pixelization
  glTranslatef(0.375, 0.375, 0);

  // Clear.
  glClearColor(.5f, 0.0f, 0.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  // What Every CUDA Programmer Should Know About OpenGL
  // http://www.nvidia.com/content/GTC/documents/1055_GTC09.pdf
  // This is using the Runtime API.

  // Allocate the GL Buffer
  // â€¢ Same as before, compute the number of bytes based upon the 
  //image data type (avoid 3 byte pixels)
  //â€¢ Do once at startup, donâ€™t reallocate unless buffer needs to grow 
  //â”€ this is expensive

  // An OpenGL buffer used for pixels and bound as GL_PIXEL_UNPACK_BUFFER 
  // is commonly called a PBO (Pixel Buffer Object)

  // Create a OpenGL Buffer(s)
  GLuint bufferID;
  // Generate a buffer ID
  glGenBuffers(1,&bufferID); 
  // Make this the current UNPACK buffer (OpenGL is state-based)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
  // Allocate data for the buffer. 4-channel 8-bit image
  glBufferData(GL_PIXEL_UNPACK_BUFFER, W * H * 4, NULL, GL_DYNAMIC_COPY);

  // Register Buffers for CUDA
  cudaGLRegisterBufferObject(bufferID);

  // Create a GL Texture
  // Enable Texturing
  glEnable(GL_TEXTURE_2D);
  // Generate a texture ID
  GLuint textureID;
  glGenTextures(1,&textureID); 
  // Make this the current texture (remember that GL is state-based)
  glBindTexture(GL_TEXTURE_2D, textureID); 
  // Allocate the texture memory. The last parameter is NULL since we only
  // want to allocate memory, not initialize it 
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
  // Must set the filter mode, GL_LINEAR enables interpolation when scaling 
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  // Note: GL_TEXTURE_RECTANGLE_ARB may be used instead of GL_TEXTURE_2D 
  // for improved performance if linear interpolation is not desired. Replace 
  // GL_LINEAR with GL_NEAREST in the glTexParameteri() call.

  //  // the texture wraps over at the edges(repeat)
  //  //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  //  //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

  state_setup(W, H);

  float k(0.0f);
  bool run = true;
  while (run) {
    SDL_Event event;
    // Process all events in the queue.
    while (SDL_PollEvent(&event)) {
      switch (event.type) {
        case SDL_KEYDOWN:
          run = false;
          break;

        default:
          break;
      }
    }

    // Map the GL Buffer to CUDA

    //Provides a CUDA pointer to the GL bufferâ”€on 
    //a single GPU no data is moved (Win & Linux)
    //â€¢ When mapped to CUDA, OpenGL should not use 
    //this buffer
    void *devPtr;
    cudaGLMapBufferObject(&devPtr, bufferID);

    //Write to the Image

    //â€¢ CUDA C kernels may now use the mapped memory 
    //just like regular GMEM
    //â€¢ CUDA copy functions can use the mapped memory 
    //as a source or destination

    //static bool flip = false;
    //flip = !flip;
    //cuda_write((int*)devPtr, W, H, (flip ? 0.0f : 1.0f));

    cuda_write((int*)devPtr, W, H, k);
    k += 0.1f;
    // Unmap the GL Buffer
    // These functions wait for all previous GPU activity to 
    // complete (asynchronous versions also available).
    cudaGLUnmapBufferObject(bufferID);

    // Create a Texture From the Buffer

    // Select the appropriate buffer 
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID); 
    // Select the appropriate texture
    glBindTexture(GL_TEXTURE_2D, textureID); 
    // Make a texture from the buffer
    // glTexSubImage2D source parameter is NULL, Data is  coming from a PBO, not host memory
    // Note: glTexSubImage2D will perform a format conversion if the buffer is a
    // different format from the texture. We created the texture with format 
    // GL_RGBA8. In glTexSubImage2D we specified GL_BGRA and GL_UNSIGNED_INT.
    // This is a fast-path combination.
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_BGRA, GL_UNSIGNED_BYTE, NULL); // orig from presentation
    //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    // Draw the image.
    // Just draw a single Quad with texture coordinates  for each vertex:
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1.0f); 
    glVertex3f(0, 0, 0);
    glTexCoord2f(0, 0);
    glVertex3f(0, H, 0);
    glTexCoord2f(1.0f, 0);
    glVertex3f(W, H, 0);
    glTexCoord2f(1.0f, 1.0f);
    glVertex3f(W, 0, 0);
    glEnd();

    SDL_GL_SwapWindow(mainwindow);

    //SDL_Delay(30);
  }

  state_destroy();

  // Unregister before freeing buffer:
  //cudaGLUnregisterBufferObject(bufferID);

  // Delete our opengl context, destroy our window, and shutdown SDL
  SDL_GL_DeleteContext(maincontext);
  SDL_DestroyWindow(mainwindow);
  SDL_Quit();

  return 0;
}
