import vtk

def get_program_parameters():
    import argparse
    description = 'Texture an object with an image.'
    epilogue = '''
   '''
    parser = argparse.ArgumentParser(description=description, epilog=epilogue,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('filename', help='masonry-wide.jpg.')
    args = parser.parse_args()
    return args.filename

ColorBackground = [0.0, 0.0, 0.0]

objpath = get_program_parameters()

reader = vtk.vtkOBJReader()

reader.SetFileName(objpath)

reader.Update()

mapper = vtk.vtkPolyDataMapper()

if vtk.VTK_MAJOR_VERSION <= 5:

     mapper.SetInput(reader.GetOutput())

else:

     mapper.SetInputConnection(reader.GetOutputPort())

actor = vtk.vtkActor()

actor.SetMapper(mapper)

# Create a rendering window and renderer

ren = vtk.vtkRenderer()

ren.SetBackground(ColorBackground)

renWin = vtk.vtkRenderWindow()

renWin.AddRenderer(ren)

# Create a renderwindowinteractor

iren = vtk.vtkRenderWindowInteractor()

iren.SetRenderWindow(renWin)

# Assign actor to the renderer

ren.AddActor(actor)

# Enable user interface interactor

iren.Initialize()

renWin.Render()

iren.Start()