"""
从GEE导出影像，分块请求，最后合并，最快导出方式

测试sentinel影像和SCL文件可使用，其它请自行测试

！！！无论影像多大都可以下，处理后的（比如计算了NDVI）可以下！！！

__author__: Polygon
__date__: 2022-10-03

2022-07-12 - 支持下载同分辨率的多波段，不同分辨率请逐个下载，重写多线程下载逻辑
2022-07-31 - 修复边界缩小bug
2022-09-03 - 修复4326导出bug；修复点读取可能出错；增加判断当前输入影响与输入分辨率是否一致
2022-09-10 - 多线程改为asyncio；加入进度条
2022-09-11 - 最后一版说明
2022-10-03 - 修复bug
"""
from datetime import datetime
import os
from pickle import NEWTRUE
from xmlrpc.client import boolean
try:
    from rich.progress import *
except:
    os.system("pip install rich")
import numpy as np
import rasterio
import asyncio
import aiohttp
import json
import ee
import io
from urllib.request import getproxies
proxies = getproxies()
if proxies:
    os.environ['HTTP_PROXY'] = proxies['http']
    os.environ['HTTPS_PROXY'] = proxies['http']
# ee.Authenticate()
print('Initialize')
ee.Initialize()
print('Success')



asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
class GEEExport:
    Array = None

    def __init__(self, ee_image_object, reference_ee_image_object=None):
        """
        比如 imageCol = [image1, image2], image1和image2都有投影信息
        image3 = (image1 + image2) / 2, 因为进行了计算丢失了投影信息
        此时, 我想下载的image3即`ee_image_object`丢失的投影信息是和image1或者image2是一样的
        称image1和image2为参考对象`reference_ee_image_object`
        """
        self.ee_image_object = ee_image_object
        self.reference_ee_image_object = reference_ee_image_object
        self.proxy = self.proxy() 
        if not asyncio._get_running_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        self.lock = asyncio.Lock()

    @staticmethod
    def proxy():
        from urllib.request import getproxies
        return getproxies()['http']

    @staticmethod
    def get_crs_transform(band):
        crs_transform = band['crs_transform']
        if 'origin' in band:
            origin = band['origin']
            print('origin', origin)
            crs_transform[2] += crs_transform[0] * origin[0]
            crs_transform[5] += crs_transform[4] * origin[1]
        return crs_transform

    @staticmethod
    def getKeyValue(d, k):
        def _getKeyValue(d, k):
            for k in list(k.split(".")):
                if k in d:
                    d = d[k]
                else:
                    return False
            return d
        if type(k) == list:
            k_list = k.copy()
            for k in k_list:
                v = _getKeyValue(d, k)
                if v: return v
            return v
        else:
            return _getKeyValue(d, k)

    async def export_small_image(self, session, params):
        """
        导出限制像素内的影像瓦片
        """
        url_params = params.copy()
        row, col = url_params.pop('index')
        try:
            url = await self.loop.run_in_executor(None, self.ee_image_object.getDownloadURL, url_params)
            res = await session.get(url, proxy=self.proxy)
            if res.headers['Content-Type'] != 'application/octet-stream': return await self.export_small_image(session, params)
            buffer = b""
            async for chunk in res.content.iter_any():
                buffer += chunk
            data = np.load(io.BytesIO(buffer))
            # 写入self.Array
            bandName_list = list(dict(data.dtype.fields).keys())
            # 判断是否需要创建self.Array
            # 这里需要锁，保证不会被多次创建
            async with self.lock:
                if self.Array is None:
                    dtype = data[bandName_list[0]].dtype
                    self.Array = np.empty(self.Array_shape, dtype=dtype)
            for i, bandName in enumerate(bandName_list):
                self.Array[i, row:row+data.shape[0], col:col+data.shape[1]] = data[bandName]
            self.finished += data.shape[0] * data.shape[1]
        except Exception as e:
            print(e)
            return await self.export_small_image(session, params)
    
    async def progress(self):
        progress_columns = (
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn(),
            RenderableColumn()
        )
        with Progress(*progress_columns) as progress_bar:
            task = progress_bar.add_task("[blue]Downloading...", total=self.total)
            finished = 0
            while True:
                advance = self.finished - finished
                if advance:
                    progress_bar.advance(task, advance=advance)
                    finished = self.finished
                else:
                    await asyncio.sleep(0)
                if finished == self.total: 
                    await asyncio.sleep(1)
                    break

    def clip(self, points):
        points = np.array(points).reshape(1, -1, 2).tolist()
        roi = ee.Geometry({
            "type": "Polygon",
            "coordinates": points,
            "geodesic": False,
            "evenOdd": True
        })
        self.ee_image_object = self.ee_image_object.clip(roi)
        info = self.ee_image_object.getInfo()
        return info

    async def export_img(self, out_dirname, geo_params, prefix, local, silence=False):
        """ 
        下载image的第一个波段
        num_workers - 为concurrent.futures.ThreadPoolExecutor所需参数，同时下载个数

        geo_params = dict(crs=None, scale=None, extent=None, points=None)
            crs    - 投影坐标系
            scale  - 分辨率，始终是米，无论crs是什么
            extent - [leftX, topY, rowN, colN]  投影坐标系下的左上角坐标和行列数
            points - [[[lon, lat], [lon, lat]]] 经纬度坐标下的矩形由这些点坐标组成

            extent和points二选一，优先级extend > points
        """
        # 检查输出文件储存目录
        if not os.path.exists(out_dirname): os.mkdir(out_dirname)
        # 获取ee.Image详细信息
        info = self.ee_image_object.getInfo()
        # 检查影像本身分辨率和crs与输入是否一直需不需要裁剪
        obj_crs = info['bands'][0]['crs']
        obj_scale = self.ee_image_object.projection().nominalScale().getInfo()
        if self.reference_ee_image_object:
            reference_info = self.reference_ee_image_object.getInfo()
            if not self.getKeyValue(geo_params, "scale"):
                geo_params["scale"] = reference_info['bands'][0]["crs_transform"][0]
            if not self.getKeyValue(geo_params, "crs"):
                geo_params["crs"] = reference_info['bands'][0]['crs']
        crs = self.getKeyValue(geo_params, "crs") or obj_crs
        scale = self.getKeyValue(geo_params, "scale") or obj_scale
        if (scale != obj_scale) or (crs != obj_crs):
            # 不一致，打印出来
            self.ee_image_object = self.ee_image_object.reproject(**{
                "crs": crs,
                "scale": scale
            })
            info = self.ee_image_object.getInfo()
            # 参考影像（若有）也需要重投影
            if self.reference_ee_image_object:
                self.reference_ee_image_object = self.reference_ee_image_object.reproject(**{
                    "crs": crs,
                    "scale": scale
                })
        if self.reference_ee_image_object:
            # 存在参考影像，把参考影像基本信息写入要下载影像（可能缺失基本信息）
            reference_band_info = reference_info["bands"][0]
            for band_info in info["bands"]:
                for key in ["dimensions", "crs", "crs_transform", "origin"]:
                    value = self.getKeyValue(reference_band_info, key)
                    if value:
                        band_info[key] = value
            info["properties"] = reference_info["properties"]
        print(geo_params)
        bandnames = [band['id'] for band in info['bands']]
        obj_name = self.getKeyValue(info, ["properties.system:index", "id"])
        out_filename = '_'.join(([prefix] if prefix else []) + ([os.path.basename(obj_name)] if obj_name else []) + ['_'.join(bandnames)])
        out_params = {"driver": "GTiff", "ext": ".tif"}
        out_filename = os.path.join(out_dirname, out_filename + out_params["ext"])
        # 检查本地
        if os.path.exists(out_filename) and local:
            print("read local")
            with rasterio.open(out_filename, 'r+') as src:
                try:
                    json.loads(src.tags()['info'])
                except:
                    src.update_tags(info=json.dumps(info))
            return os.path.join(os.getcwd(), out_filename)
        # 获取crs_transform和dimensions参数
        crs_transform = None
        dimensions = None
        # 判断输入是否包含裁剪参数
        if self.getKeyValue(geo_params, ["extent", "points"]):
            # 有裁剪参数，必须是输入的crs和scale下的extent
            extent = self.getKeyValue(geo_params, "extent")
            points = self.getKeyValue(geo_params, "points")
            if extent:
                # 规定了extent可以解析出可用于下载的参数，无需多余操作
                print("extent")
                crs_transform = [
                    scale, 0, int(extent[0]),
                    0, -scale, int(extent[1])
                ]
                dimensions = [int(extent[-1]), int(extent[-2])]
            elif points:
                print("points")
                info = self.clip(points)
                crs_transform = self.get_crs_transform(info['bands'][0])
                dimensions = info['bands'][0]['dimensions']
        else:
            # 没有裁剪参数
            # 尝试直接直接读取
            invalid_crs_transform = [1, 0, 0, 0, 1, 0]
            if set({'dimensions', 'crs_transform'}) <= set(info['bands'][0].keys()) and info['bands'][0]["crs_transform"] != invalid_crs_transform:
                # 这里有效的crs_transform和dimensions可能继承自reference_ee_image_object
                print("read from ee_image_object")
                crs_transform = self.get_crs_transform(info['bands'][0])
                dimensions = info['bands'][0]['dimensions']
            else:
                # 没有有效的crs_transform和dimensions
                print(info['bands'][0]["crs_transform"])
                footprint_points = self.getKeyValue(info, 'properties.system:footprint.coordinates')
                if footprint_points:
                    print("footprint_points")
                    info = self.clip(footprint_points)
                    crs_transform = self.get_crs_transform(info['bands'][0])
                    dimensions = info['bands'][0]['dimensions']
                else:
                    raise ValueError(f"crs={crs}, geo_params={geo_params}, dimensions={dimensions}, crs_transform={crs_transform}\n输入影像丢失投影信息，需要指定points或reference_ee_image_object其中之一")
        print(scale, crs_transform, dimensions)
        # 储存变量
        self.Array_shape = (len(bandnames), dimensions[1], dimensions[0])
        # 分配任务
        max_bytes = 50331648
        # 单数据所占字节
        bytesN = 100
        offset = int((max_bytes / bytesN / len(bandnames)) ** .5)
        offset = 512
        # 下面需要优化，避免最后的rasterio的merge，很耗费
        # sss - start, stop, step
        x_start_stop_step = [crs_transform[2], crs_transform[2] + dimensions[0] * crs_transform[0], offset * crs_transform[0]]
        y_start_stop_step = [crs_transform[5], crs_transform[5] + dimensions[1] * crs_transform[4], offset * crs_transform[4]]
        # 储存要处理的起始点
        x_list = np.arange(*x_start_stop_step).tolist()
        y_list = np.arange(*y_start_stop_step).tolist()
        end_x = crs_transform[2] + dimensions[0]*crs_transform[0]
        end_y = crs_transform[5] + dimensions[1]*crs_transform[4]
        self.total = dimensions[0] * dimensions[1]
        self.finished = 0
        """
          (x_list[1], y_list[0])   
                    ↓      
                +   +   +   |
                            |
                +   +   +   |
                            |
                +   +   +   |
                ____________|
        """
        # 生成所有的参数
        params_list = []
        for col, x in enumerate(x_list):
            for row, y in enumerate(y_list):
                # 计算小块的dimensions
                _dimensions = [offset, offset]
                if y + y_start_stop_step[-1] < y_start_stop_step[1]:
                    _dimensions[1] = int((end_y - y) / crs_transform[4])
                if x + x_start_stop_step[2] > x_start_stop_step[1]:
                    _dimensions[0] = int((end_x - x) / crs_transform[0])
                # 临时文件，不储存临时文件
                # temp_filename = f'{temp}/{_out_filename}_part_{x}_{y}.tif'
                _crs_transform = crs_transform.copy()
                _crs_transform[2] = x
                _crs_transform[5] = y
                index = (row*offset, col*offset)
                params = {
                    "name": str(index), 
                    "index": index,
                    "filePerBand": False,
                    "scale": scale,
                    "dimensions": _dimensions,
                    "crs": crs,
                    "crs_transform": _crs_transform,
                    "format": 'NPY',
                }
                params_list.append(params)
        self.loop = asyncio.get_event_loop()
        tasks = []
        if not silence: tasks.append(self.progress())
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(sock_read=5)) as session:
            tasks += [self.loop.create_task(self.export_small_image(session, params)) for params in params_list]
            await asyncio.gather(*tasks)
        # 写入本地文件
        print('\nWrite to local...')
        async def toLocal(out_filename):
            profile = {
                "width": self.Array.shape[-1],
                "height": self.Array.shape[-2],
                "count": self.Array.shape[0],
                "driver": out_params["driver"],
                "crs": crs,
                "transform": crs_transform,
                "dtype": self.Array.dtype
            }
            # 可能文件占用
            try:
                with rasterio.open(out_filename, "w", **profile) as dst:
                    dst.update_tags(info=json.dumps(info))
                    for i in range(profile["count"]):
                        dst.write(self.Array[i], i+1)
                return out_filename
            except Exception as e:
                print(e)
                print("add time to out_filename")
                # 给out_filename加上时间参数
                out_filename = os.path.join(
                    os.path.dirname(out_filename), 
                    os.path.splitext(os.path.basename(out_filename))[0] + datetime.now().strftime("%Y%m%dT%H%M%S") + out_params["ext"]
                )
                await asyncio.sleep(1)
                return await toLocal(out_filename)  
        out_filename = await toLocal(out_filename)
        print('Finished')
        return os.path.join(os.getcwd(), out_filename)


if __name__ == "__main__":
    # image = (ee.ImageCollection("COPERNICUS/S2")
    #     .filterDate(ee.Date('2021-10-19').getRange('month'))
    #     .filterMetadata('system:index', 'contains', 'T50TMK')
    #     .select('B2', "B3", "B4")
    #     .first()
    # )
    # filename = asyncio.run(
    #     GEEExport(
    #         ee_image_object=image
    #         ).export_img(
    #             'output',
    #             geo_params=dict(crs=None, scale=None, extent=None, points=None),
    #             prefix='test',
    #             local=False
    #         )
    # )
    # print(filename)
    day = '2021-12-05'
    regionId = 'T50TMK'
    imageCol = (ee.ImageCollection("COPERNICUS/S2")
        .filterDate(ee.Date(day).getRange('month'))
        .filterMetadata('system:index', 'contains', regionId)
    )
    p = 20
    image = (imageCol
        .reduce(ee.Reducer.percentile([p]))
    )

    filename = asyncio.run(
        GEEExport(
            ee_image_object=image.select(f"B1_p{p}"),
            reference_ee_image_object=imageCol.first().select("B1")
            ).export_img(
                'output',
                geo_params=dict(crs=None, scale=None, extent=None, points=None),
                prefix='p20',
                local=False
            )
    )
