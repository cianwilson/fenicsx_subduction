
import numpy
from scipy import interpolate as interp
from scipy import integrate as integ
from scipy import optimize as opt

class ElementaryEntity:
  def __init__(self, name=None):
    self.name = name
    self.eid  = None

  def cleareid(self):
    self.eid = None

class PhysicalEntity:
  def __init__(self, pid=None):
    self.pid = pid

class Point(ElementaryEntity, PhysicalEntity):
  def __init__(self, coord, name=None, res=1.0):
    self.type = "Point"
    self.x = coord[0]
    if len(coord) > 1: 
      self.y = coord[1]
    else:
      self.y = 0.0
    if len(coord) > 2: 
      self.z = coord[2]
    else:
      self.z = 0.0
    self.res = res
    ElementaryEntity.__init__(self, name=name)
    PhysicalEntity.__init__(self)

class Curve(ElementaryEntity, PhysicalEntity):
  def __init__(self, points, name=None, pid=None):
    self.points = points
    self.name = name
    self.x = None
    self.y = None
    self.u = None
    ElementaryEntity.__init__(self, name=name)
    PhysicalEntity.__init__(self, pid=pid)

  def update(self):
    self.x = numpy.array([self.points[i].x for i in range(len(self.points))])
    isort = numpy.argsort(self.x)
    points = numpy.asarray(self.points)[isort]
    self.points = points.tolist()
    self.x = numpy.array([self.points[i].x for i in range(len(self.points))])
    self.y = numpy.array([self.points[i].y for i in range(len(self.points))])

  def cleareid(self):
    for point in self.points: point.cleareid()
    ElementaryEntity.cleareid(self)

class Line(Curve):
  def __init__(self, points, name=None, pid=None):
    assert(len(points)==2)
    self.type = "Line"
    Curve.__init__(self, points, name=name, pid=pid)
    self.update()

  def update(self):
    Curve.update(self)
    self.u = numpy.arange(0.0, self.x.size)

  def __call__(self, u):
    return [self.points[0].x + u*(self.points[1].x - self.points[0].x), \
            self.points[0].y + u*(self.points[1].y - self.points[0].y)]

class InterpolatedCubicSpline:
  def __init__(self, points, name=None, pids=None, bctype='natural'):
    self.type = "InterpolatedCubicSpline"
    self.bctype = bctype
    self.points = [point for point in points]
    self.controlpoints = [point for point in points]
    self.controlpoints.sort(key=lambda point: point.x)
    self.name = name
    if pids is not None: 
      assert(len(pids)==len(self.points)-1)
      self.pids = pids
    else:
      self.pids = [None for i in range(len(self.points)-1)]
    self.x = None
    self.y = None
    self.u = None
    self.interpu = None
    self.interpcurves = None
    self.length = None
    self.update()

  def __call__(self, delu, x0=None, der=0):
    # NOTE: this returns [x, dy/dx] when der=1...
    x = self.delu2x(delu, x0=x0)
    return [x, float(self.cs(x, nu=der))]

  def x2delu(self, x, x0=None):
    """Convert from Delta x to Delta u:
       u = \int_{x0}^{x} sqrt(1 + (dy(x')/dx')**2) dx'
       x0 is the lower bound of integration - provide to get an incremental u"""
    if x0 is None: x0 = self.x[0]
    return integ.quad(lambda xp: self.du(xp), x0, x)[0]/self.length

  def delu2x(self, delu, x0=None):
    """Convert from u to x:
       u(x) = \int_{x0}^x sqrt(1 + (dy(x')/dx')**2) dx'
       x0 is the lower bound of integration."""
    if x0 is None: x0 = self.x[0]
    return opt.fsolve(lambda x: self.x2delu(x, x0=x0)-delu, x0, fprime=lambda x: [self.du(x)])[0]

  def du(self, x):
    return numpy.sqrt(1.+float(self.cs(x, nu=1))**2)

  def update(self):
    self.x = numpy.array([self.points[i].x for i in range(len(self.points))])
    isort = numpy.argsort(self.x)
    points = numpy.asarray(self.points)[isort]
    self.points = points.tolist()
    pids = numpy.asarray(self.pids)[[i for i in isort if i != (len(isort)-1)]]
    self.pids = pids.tolist()
    self.x = numpy.array([self.points[i].x for i in range(len(self.points))])
    self.y = numpy.array([self.points[i].y for i in range(len(self.points))])
    controlx = numpy.array([cp.x for cp in self.controlpoints])
    controly = numpy.array([cp.y for cp in self.controlpoints])
    self.cs = interp.CubicSpline(controlx, controly, bc_type=self.bctype)
    self.length = 1.0 # must set this to one first to get the next line correct
    self.length = self.x2delu(self.x[-1])
    u = numpy.asarray([0.0])
    u = numpy.append(u, [self.x2delu(self.x[i], x0=self.x[i-1]) for i in range(1,len(self.x))])
    self.u = numpy.cumsum(u)
  
  def updateinterp(self):
    self.interpu = []
    self.interpcurves = []
    for p in range(len(self.points)-1):
      pid = self.pids[p]
      lengthfraction = self.u[p+1]-self.u[p]
      res0 = self.points[p].res/self.length/lengthfraction
      res1 = self.points[p+1].res/self.length/lengthfraction
      t = 0.0
      ts = [t]
      while t < 1.0:
        t = ts[-1] + (1.0 - t)*res0 + t*res1
        ts.append(t)
      ts = numpy.array(ts)/ts[-1]
      ls = (ts[1:]-ts[:-1])*lengthfraction*self.length
      res = [max(ls[i], ls[i+1]) for i in range(len(ls)-1)]
      point = self.points[p]
      self.interpu.append(self.u[p])
      for i in range(1,len(ts)-1):
        t = ts[i]
        self.interpu.append(self.u[p] + t*lengthfraction)
        npoint = Point(self(t*lengthfraction, x0=self.points[p].x), res=res[i-1])
        self.interpcurves.append(Line([point, npoint], pid))
        point = npoint
      npoint = self.points[p+1]
      self.interpcurves.append(Line([point, npoint], pid))
    self.interpu.append(self.u[p+1])
    self.interpu = numpy.asarray(self.interpu)

  def updateresolutions(self):
    res0 = self.points[0].res
    u0   = self.u[0]
    for p in range(len(self.points)):
      point = self.points[p]
      if point.res is None:
        for q in range(p, len(self.points)):
          pointq = self.points[q]
          if pointq.res is not None: break
        res1 = pointq.res
        u1   = self.u[q]
        t = (self.u[p] - u0)/(u1 - u0)
        res = (1.-t)*res0 + t*res1
        point.res = res
      else:
        res0 = point.res
        u0   = self.u[p]

  def updateinterpids(self, pid, point0, point1):
    for curve in self.interpocurvesinterval(self.findpoint(point0), self.findpoint(point1)):
      curve.pid = pid


  #def copyinterp(self, spline):
  #  assert(len(self.points)==len(spline.points))
  #  self.interpu = []
  #  self.interpcurves = []
  #  for p in range(len(self.points)-1):
  #    pid = self.pids[p]
  #    lengthfraction = self.u[p+1]-self.u[p]
  #    splinelengthfraction = spline.u[p+1]-spline.u[p]
  #    lengthratio = lengthfraction/splinelengthfraction
  #    splinepoint0 = spline.points[p]
  #    splinepoint1 = spline.points[p+1]
  #    splineus = spline.interpusinterval(splinepoint0, splinepoint1)
  #    splinecurves = spline.interpcurvesinterval(splinepoint0, splinepoint1)
  #    assert(len(splineus)==len(splinecurves)+1)
  #    point = self.points[p]
  #    self.interpu.append(self.u[p])
  #    for i in range(1, len(splineus)-1):
  #      self.interpu.append(self.u[p] + (splineus[i]-spline.u[p])*lengthratio)
  #      npoint = Point(self((splineus[i]-spline.u[p])*lengthratio, x0=self.points[p].x), \
  #                     res=splinecurves[i-1].points[1].res*lengthratio)
  #      self.interpcurves.append(Line([point,npoint], pid))
  #      point = npoint
  #    npoint = self.points[p+1]
  #    self.interpcurves.append(Line([point, npoint], pid))
  #  self.interpu.append(self.u[p+1])
  #  self.interpu = numpy.asarray(self.interpu)

  def intersecty(self, yint, extrapolate=False):
    return [self.cs.solve(yint, extrapolate=extrapolate)[0], yint]
      
  def intersectx(self, xint):
    return [xint, float(self.cs(xint))]

  def interpcurveindex(self, u):
    loc = abs(self.interpu - u).argmin()
    if loc == 0: 
      loc0 = loc
    elif loc == len(self.interpu)-1: 
      loc0 = loc-1
    else:
      if self.interpu[loc] < u: 
        loc0 = loc
      else: 
        loc0 = loc-1
    return loc0

  def unittangentx(self, x):
    der = self.cs(x, nu=1)
    vec = numpy.array([1.0, der])
    vec = vec/numpy.sqrt(sum(vec**2))
    return vec

  def translatenormal(self, dist):
    for p in range(len(self.points)):
      der = float(self.cs(self.points[p].x, nu=1))
      vec = numpy.array([-der, 1.0])
      vec = vec/numpy.sqrt(sum(vec**2))
      self.points[p].x += dist*vec[0]
      self.points[p].y += dist*vec[1]
    self.update()

  def croppoint(self, pind, coord):
    for i in range(len(pind)-1):
      p = self.points.pop(pind[i])
      if p in self.controlpoints:
        self.controlpoints.pop(self.controlpoints.index(p))
    self.points[pind[-1]].x = coord[0]
    self.points[pind[-1]].y = coord[1]
    self.update()

  def crop(self, left=None, bottom=None, right=None, top=None):
    if left is not None:
      out = numpy.where(self.x < left)[0]
      if (len(out)>0):
        coord = self.intersectx(left)
        self.croppoint(out, coord)
    if right is not None:
      out = numpy.where(self.x > right)[0]
      if (len(out)>0):
        coord = self.intersectx(right)
        self.croppoint(out, coord)
    if bottom is not None:
      out = numpy.where(self.y < bottom)[0]
      if (len(out)>0):
        coord = self.intersecty(bottom)
        self.croppoint(out, coord)
    if top is not None:
      out = numpy.where(self.y > top)[0]
      if (len(out)>0):
        coord = self.intersecty(top)
        self.croppoint(out, coord)

  def translatenormalandcrop(self, dist):
    left = self.x.min()
    right = self.x.max()
    bottom = self.y.min()
    top = self.y.max()
    self.translatenormal(dist)
    self.crop(left=left, bottom=bottom, right=right, top=top)

  def findpointx(self, xint):
    ind = numpy.where(self.x==xint)[0]
    if (len(ind)==0): 
      return None
    else:
      return self.points[ind[0]]
  
  def findpointy(self, yint):
    ind = numpy.where(self.y==yint)[0]
    if (len(ind)==0): 
      return None
    else:
      return self.points[ind[0]]
  
  def findpoint(self, name):
    for p in self.points:
      if p.name == name: return p
    return None

  def findpointindex(self, name):
    for p in range(len(self.points)):
      if self.points[p].name == name: return p
    return None

  def interpcurvesinterval(self, point0, point1):
    for l0 in range(len(self.interpcurves)):
      if self.interpcurves[l0].points[0] == point0: break

    for l1 in range(l0, len(self.interpcurves)):
      if self.interpcurves[l1].points[1] == point1: break

    return self.interpcurves[l0:l1+1]

  def interpusinterval(self, point0, point1):
    for l0 in range(len(self.interpcurves)):
      if self.interpcurves[l0].points[0] == point0: break

    for l1 in range(l0, len(self.interpcurves)):
      if self.interpcurves[l1].points[1] == point1: break

    return self.interpu[l0:l1+2]

  def appendpoint(self, p, pid=None):
    self.points.append(p)
    self.pids.append(pid)
    self.update()

  def cleareid(self):
    for curve in self.interpcurves: curve.cleareid()

class Surface(ElementaryEntity, PhysicalEntity):
  def __init__(self, curves, name=None, pid = None):
    self.type = "Surface"
    self.curves = [curves[0]]
    self.directions = [1]
    ind = list(range(1, len(curves)))
    for j in range(1, len(curves)):
      pcurve = self.curves[j-1]
      if self.directions[j-1]==-1:
        pind = 0
      else:
        pind = -1
      for i in range(len(ind)):
        if pcurve.points[pind] == curves[ind[i]].points[0]:
          self.directions.append(1)
          break
        elif pcurve.points[pind] == curves[ind[i]].points[-1]:
          self.directions.append(-1)
          break
      if i == len(ind)-1 and len(self.directions) == j:
        print("Failed to complete line loop.")
        sys.exit(1)
      self.curves.append(curves[ind[i]])
      i = ind.pop(i)
    ElementaryEntity.__init__(self, name=name)
    PhysicalEntity.__init__(self, pid=pid)

  def cleareid(self):
    for curve in self.curves: curve.cleareid()
    ElementaryEntity.cleareid(self)

class Geometry:
  def __init__(self):
    self.namedpoints = {}
    self.namedcurves = {}
    self.namedinterpcurves = {}
    self.namedsurfaces = {}
    self.physicalpoints = {}
    self.physicalcurves = {}
    self.physicalsurfaces = {}
    self.points  = []
    self.curves  = []
    self.interpcurves  = []
    self.surfaces = []
    self.pointcomments  = []
    self.curvecomments  = []
    self.interpcurvecomments  = []
    self.surfacecomments = []
    self.pointembeds = {}
    self.lineembeds = {}
    self.transfinitecurves = {}
    self.transfinitesurfaces = {}
    self.geofile = None

  def addpoint(self, point, name=None, comment=None):
    self.points.append(point)
    self.pointcomments.append(comment)
    if not comment:
      if name:
        self.pointcomments[-1] = name
      elif point.name:
        self.pointcomments[-1] = point.name
    if name:
      self.namedpoints[name] = point
    elif point.name:
      self.namedpoints[point.name] = point
    if point.pid:
      if point.pid in self.physicalpoints:
        self.physicalpoints[point.pid].append(point)
      else:
        self.physicalpoints[point.pid] = [point]

  def addcurve(self, curve, name=None, comment=None):
    self.curves.append(curve)
    self.curvecomments.append(comment)
    if not comment:
      if name:
        self.curvecomments[-1] = name
      elif curve.name:
        self.curvecomments[-1] = curve.name
    if name:
      self.namedcurves[name] = curve
    elif curve.name:
      self.namedcurves[curve.name] = curve
    if curve.pid:
      if curve.pid in self.physicalcurves:
        self.physicalcurves[curve.pid].append(curve)
      else:
        self.physicalcurves[curve.pid] = [curve]

  def addtransfinitecurve(self, curve, name=None, comment=None, n=2):
    self.addcurve(curve, name, comment)
    if n in self.transfinitecurves:
      self.transfinitecurves[n].append(curve)
    else:
      self.transfinitecurves[n] = [curve]

  def addtransfinitesurface(self, surface, corners, direction="Right", name=None, comment=None):
    self.addsurface(surface, name, comment)
    self.transfinitesurfaces[surface] = (corners, direction)

  def addinterpcurve(self, interpcurve, name=None, comment=None):
    self.interpcurves.append(interpcurve)
    self.interpcurvecomments.append(comment)
    if not comment:
      if name:
        self.interpcurvecomments[-1] = name
      elif interpcurve.name:
        self.interpcurvecomments[-1] = interpcurve.name
    if name:
      self.namedinterpcurves[name] = interpcurve
    elif interpcurve.name:
      self.namedinterpcurves[interpcurve.name] = interpcurve
    for curve in interpcurve.interpcurves: self.addtransfinitecurve(curve)

  def addsurface(self, surface, name=None, comment=None):
    self.surfaces.append(surface)
    self.surfacecomments.append(comment)
    if not comment:
      if name:
        self.surfacecomments[-1] = name
      elif surface.name:
        self.surfacecomments[-1] = surface.name
    if name:
      self.namedsurfaces[name] = surface
    elif surface.name:
      self.namedsurfaces[surface.name] = surface
    if surface.pid:
      if surface.pid in self.physicalsurfaces:
        self.physicalsurfaces[surface.pid].append(surface)
      else:
        self.physicalsurfaces[surface.pid] = [surface]

  def addembed(self, surface, item):
    assert(surface.type=="Surface")
    assert(item.type=="Point" or item.type=="Line")
    if item.type=="Point":
      if surface in self.pointembeds:
        self.pointembeds[surface].append(item)
      else:
        self.pointembeds[surface] = [item]
    else:
      if surface in self.lineembeds:
        self.lineembeds[surface].append(item)
      else:
        self.lineembeds[surface] = [item]

  def writegeofile(self, filename):
    self.cleareid()
    
    self.geofile = GeoFile()
    for s in range(len(self.surfaces)):
      surface = self.surfaces[s]
      self.geofile.addsurface(surface, self.surfacecomments[s])
    self.geofile.linebreak()
    for c in range(len(self.curves)):
      curve = self.curves[c]
      self.geofile.addcurve(curve, self.curvecomments[c])
    self.geofile.linebreak()
    for p in range(len(self.points)):
      point = self.points[p]
      self.geofile.addpoint(point, self.pointcomments[p])
    self.geofile.linebreak()

    for surface,items in iter(sorted(list(self.pointembeds.items()), key=lambda item: item[0].eid)):
      self.geofile.addembed(surface,items)
    for surface,items in iter(sorted(list(self.lineembeds.items()), key=lambda item: item[0].eid)):
      self.geofile.addembed(surface,items)
    self.geofile.linebreak()

    for n,curves in iter(sorted(self.transfinitecurves.items())):
      self.geofile.addtransfinitecurve(curves,n)
    self.geofile.linebreak()
    for surface,corners in iter(sorted(list(self.transfinitesurfaces.items()), key=lambda item: item[0].eid)):
      self.geofile.addtransfinitesurface(surface,corners[0],corners[1])
    self.geofile.linebreak()
    
    for pid,surfaces in iter(sorted(self.physicalsurfaces.items())):
      self.geofile.addphysicalsurface(pid,surfaces)
    self.geofile.linebreak()
    for pid,curves in iter(sorted(self.physicalcurves.items())):
      self.geofile.addphysicalline(pid,curves)
    self.geofile.linebreak()
    for pid,points in iter(sorted(self.physicalpoints.items())):
      self.geofile.addphysicalpoint(pid,points)
    self.geofile.linebreak()

    self.geofile.produced_comment()

    self.geofile.write(filename)

  def pylabplot(self, lineres=100):
    import pylab
    #for interpcurve in self.interpcurves:
    #  unew = numpy.arange(interpcurve.u[0], interpcurve.u[-1]+((interpcurve.u[-1]-interpcurve.u[0])/(2.*lineres)), 1./lineres)
    #  unewx = numpy.zeros_like(unew)
    #  unewy = numpy.zeros_like(unew)
    #  for delu in unew:
    #    xy = interpcurve(delu)
    #    unewx = xy[0]
    #    unewy = xy[1]
    #  #pylab.plot(unewx, unewy, 'b')
    #  #pylab.plot(interpcurve.x, interpcurve.y, 'ob')
    #  for curve in interpcurve.interpcurves:
    #    unew = numpy.arange(curve.u[0], curve.u[-1]+((curve.u[-1]-curve.u[0])/(2.*lineres)), 1./lineres)
    #    pylab.plot(curve(unew)[0], curve(unew)[1], 'k')
    #    #pylab.plot(curve.x, curve.y, '+k')
    for curve in self.curves:
      unew = numpy.arange(curve.u[0], curve.u[-1]+((curve.u[-1]-curve.u[0])/(2.*lineres)), 1./lineres)
      pylab.plot(curve(unew)[0], curve(unew)[1])
      pylab.plot(curve.x, curve.y, 'ok')
    #for point in self.points:
    #  pylab.plot(point.x, point.y, 'ok')

  def cleareid(self):
    for surface in self.surfaces: surface.cleareid()
    for interpcurve in self.interpcurves: interpcurve.cleareid()
    for curve in self.curves: curve.cleareid()
    for point in self.points: point.cleareid()
  
class SlabSpline(InterpolatedCubicSpline):
  def __init__(self, xs, ys, name=None, pids=None, bctype='natural'):
    assert(len(xs)==len(ys))
    points = [Point((xs[i], ys[i])) for i in range(len(xs))]
    super(SlabSpline, self).__init__(points, name=name, pids=pids, bctype=bctype)

class SubductionGeometry:
  slab_spline = None

  coast_distance      = 0.0 
  extra_width         = 0.0 
  slab_side_sid       = None 
  wedge_side_sid      = None
  slab_base_sid       = None
  wedge_base_sid      = None
  coast_sid           = None
  top_sid             = None
  slab_rid             = None
  wedge_rid            = None
  coast_res           = None
  slab_side_base_res  = None
  wedge_side_top_res  = None
  wedge_side_base_res = None

  slab_side_lines   = []
  wedge_side_lines  = []
  domain_base_lines = []
  domain_top_lines  = []

  crustal_layers    = {}
  crustal_lines     = []

  wedge_side_points = {}

  slab_surfaces  = []
  wedge_surfaces = []

  def __init__(self, slab_spline, **kwargs):
    self.slab_spline = slab_spline
    self.update(**kwargs)

  def update(self, **kwargs):
    """
    Update the subduction geometry with the current values of:
    * coast_distance (>=0): distance from the trench that the coast is located,
                            does nothing if ==0.0 and the trench is at 0.0 depth
    * extra_width (>=0):    extra width of domain beyond the final slab point,
                            does nothing if ==0.0
    * slab_side_sid:        surface id of vertical side on slab side of domain
    * wedge_side_sid:       surface id of vertical side on wedge side of domain
    * slab_base_sid:        surface id of base of slab region of domain
    * wedge_base_sid:       surface id of base of wedge region of domain,
                            only used if extra_width > 0.0
    * coast_sid:            surface id of line between trench and coast,
                            only used if coast_distance > 0.0 or the slab trench is not at 0.0 depth
    * top_sid:              surface id of top of the domain
    * slab_rid:              region id of slab region
    * wedge_rid:             region id of wedge region
    * coast_res:            resolution of coast point
    * slab_side_base_res:   resolution of base of slab vertical side
    * wedge_side_top_res:   resolution of top of wedge vertical side
    * wedge_side_base_res:  resoltuion of base of wedge vertical side
    """

    # loop over kwargs setting attributes
    for k,v in kwargs.items():
      if hasattr(self, k):
        setattr(self, k, v)
    
    # first slab point
    slab_point_0 = self.slab_spline.points[0]
    slab_point_0.name = "Slab::Trench"

    # final slab point
    slab_point_f = self.slab_spline.points[-1]
    slab_point_f.name = "Slab::Base"

    self.slab_left_right = slab_point_f.x > slab_point_0.x

    # two special cases need fixing depending on if the slab goes from left to right (or not)
    # coast distance:
    coast_distance = kwargs.get('coast_distance', None)
    if coast_distance is not None: self.coast_distance = abs(coast_distance) if self.slab_left_right else -abs(coast_distance)
    # extra width:
    extra_width = kwargs.get('extra_width', None)
    if extra_width is not None: self.extra_width = abs(extra_width) if self.slab_left_right else -abs(extra_width)

    # reset the domain boundary lines
    self.slab_side_lines   = []
    self.wedge_side_lines  = []
    self.domain_base_lines = []
    self.domain_top_lines  = []
    self.crustal_lines     = []
    
    self.trench_x = slab_point_0.x
    self.trench_y = slab_point_0.y
    if self.trench_y > 0.0:
      raise Exception("First point in slab ({}, {}) must be at or below domain surface.".format(slab_point_0.x, slab_point_0.y))

    self.wedge_side_x  = slab_point_f.x + self.extra_width
    self.domain_base_y = slab_point_f.y

    # set up a lower slab side point
    slab_side_base = Point([self.trench_x, self.domain_base_y], "SlabSide::Base", res=self.slab_side_base_res)
    # use it to make the slab side line up to the trench
    self.slab_side_lines.append(Line([slab_side_base, slab_point_0], name="SlabSide", pid=self.slab_side_sid))

    # define the wedge side base point, which may be the final slab point
    wedge_side_base = slab_point_f
    # or a completely new point
    if abs(self.extra_width) > 0.0:
      wedge_side_base = Point([self.wedge_side_x, self.domain_base_y], "WedgeSide::Base", res=self.wedge_side_base_res)
    # define an upper wedge side point
    wedge_side_top = Point([self.wedge_side_x, 0.0], "WedgeSide::Top", res=self.wedge_side_top_res)
    # define the wedge side line
    self.wedge_side_lines.append(Line([wedge_side_base, wedge_side_top], name="WedgeSide", pid=self.wedge_side_sid))
    
    # start on the lower lines
    # these go up to the slab base
    self.domain_base_lines.append(Line([slab_side_base, slab_point_f], name="SlabBase", pid=self.slab_base_sid))
    # but continue if extra width is > 0.0
    if self.extra_width > 0.0:
      self.domain_base_lines.append(Line([slab_point_f, wedge_side_base], name="WedgeBase", pid=self.wedge_base_sid))

    coast_point = slab_point_0
    if self.coast_distance > 0.0 or self.trench_y < 0.0:
      coast_point = Point([self.trench_x+self.coast_distance, 0.0], "WedgeTop::Coast", res=self.coast_res)
      self.domain_top_lines.append(Line([slab_point_0, coast_point], name="WedgeCoast", pid=self.coast_sid))
    self.domain_top_lines.append(Line([coast_point, wedge_side_top], "WedgeTop", pid=self.top_sid))

    self.sortlines()

    for name, layer_dict in  self.crustal_layers.items():
      self.addcrustlayer(layer_dict["depth"], name, layer_dict["pid"], layer_dict["slab_res"], layer_dict["side_res"])

    for name, point_dict in self.wedge_side_points.items():
      self.addwedgesidepoint(point_dict["depth"], name, point_dict["line_name"], point_dict["res"])

  def sortlines(self):
    for lines in [self.slab_side_lines, self.wedge_side_lines, self.crustal_lines]:
      lines.sort(key=lambda line: line.y.min())
    for lines in [self.domain_base_lines, self.domain_top_lines]:
      lines.sort(key=lambda line: line.x[0])

  def addcrustlayer(self, depth, name, pid=None, slab_res=1.0, side_res=1.0):
    """
    Add a crustal layer to the subduction geometry.
    """
    if depth < 0.0: depth = -depth
    self.crustal_layers[name] = {
                                  "depth"    : depth,
                                  "pid"      : pid,
                                  "slab_res" : slab_res,
                                  "side_res" : side_res,
                                }
    point0 = self.addslabpoint(depth, "Slab::"+name, res=slab_res)
    point1 = self.addwedgesidepoint(depth, "WedgeSide::"+name, line_name=name, res=side_res)
    line = Line([point0, point1], name=name, pid=pid)
    self.crustal_lines.append(line)
    self.sortlines()
    return line

  def addslabpoint(self, depth, name, res=None):
    if depth < 0.0: depth = -depth
    if depth < -self.trench_y or depth > -self.domain_base_y:
      raise Exception("Depth {} must be within range of slab depths ({}, {}).".format(depth, -self.trench_y, -self.domain_base_y))
    point0 = self.slab_spline.findpointy(-depth)
    if point0 is not None:
      point0.name = name
      if res is not None: point0.res  = res
    else:
      point0 = Point(self.slab_spline.intersecty(-depth), name=name, res=res)
      self.slab_spline.appendpoint(point0)
      self.slab_spline.updateinterp()
    return point0

  def addwedgesidepoint(self, depth, name, line_name=None, res=None):
    if depth < 0.0: depth = -depth
    if line_name is None: line_name = name
    self.wedge_side_points[name] = {
                                     "depth" : depth,
                                     "res"   : res,
                                     "line_name" : line_name,
                                   }
    sidepoints  = [line.points[0] for line in self.wedge_side_lines]
    sidepoints += [self.wedge_side_lines[-1].points[1]]
    points = [point for point in sidepoints if point.y==-depth]
    if len(points) > 0:
      point = points[0]
      if res is not None: point.res = res
    else:
      point = Point([self.wedge_side_x, -depth], name, res=res)
      for i, line in enumerate(self.wedge_side_lines):
        if line.y.min() <= -depth <= line.y.max(): break
      iymin = line.y.argmin()
      self.wedge_side_lines[i] = Line([line.points[iymin], point], name=line.name)
      self.wedge_side_lines.insert(i+1, Line([point, line.points[iymin-1]], name=line_name))
      self.sortlines()
    return point

  def plot(self):
    geom = Geometry()
    if self.slab_spline is not None: geom.addinterpcurve(self.slab_spline)
    for lines in [self.slab_side_lines, self.wedge_side_lines, self.domain_base_lines, self.domain_top_lines, self.crustal_lines]:
      for line in lines:
        geom.addcurve(line)
    geom.pylabplot()
