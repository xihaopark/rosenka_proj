"""
hierarchical_navigator.py
分层文件系统导航器
"""

class HierarchicalNavigator:
    """基于文件夹结构的分层导航"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.index = self._build_index()
        
    def _build_index(self):
        """构建层级索引"""
        index = {
            'prefectures': {},
            'cities': {},
            'districts': {},
            'files': {}
        }
        
        # 扫描并构建索引
        for pref_path in self.data_dir.iterdir():
            if not pref_path.is_dir():
                continue
                
            pref_name = pref_path.name
            index['prefectures'][pref_name] = {
                'path': pref_path,
                'cities': []
            }
            
            for city_path in pref_path.iterdir():
                if not city_path.is_dir():
                    continue
                    
                city_name = city_path.name
                index['cities'][f"{pref_name}/{city_name}"] = {
                    'path': city_path,
                    'districts': []
                }
                index['prefectures'][pref_name]['cities'].append(city_name)
                
                for district_path in city_path.iterdir():
                    if not district_path.is_dir():
                        continue
                        
                    district_name = district_path.name
                    full_path = f"{pref_name}/{city_name}/{district_name}"
                    
                    pdf_files = list(district_path.glob("*.pdf"))
                    index['districts'][full_path] = {
                        'path': district_path,
                        'files': pdf_files
                    }
                    index['cities'][f"{pref_name}/{city_name}"]['districts'].append(district_name)
                    
                    # 索引每个文件
                    for pdf_file in pdf_files:
                        index['files'][str(pdf_file)] = {
                            'prefecture': pref_name,
                            'city': city_name,
                            'district': district_name,
                            'full_path': full_path
                        }
        
        return index
    
    def navigate_to_files(self, address_components: Dict[str, str]) -> List[Path]:
        """根据地址组件导航到文件"""
        prefecture = address_components.get('prefecture', '')
        city = address_components.get('city', '')
        district = address_components.get('district', '')
        
        # 构建查询路径
        if district:
            query_path = f"{prefecture}/{city}/{district}"
            district_info = self.index['districts'].get(query_path, {})
            return district_info.get('files', [])
        elif city:
            query_path = f"{prefecture}/{city}"
            city_info = self.index['cities'].get(query_path, {})
            files = []
            for dist in city_info.get('districts', []):
                dist_path = f"{query_path}/{dist}"
                files.extend(self.index['districts'].get(dist_path, {}).get('files', []))
            return files
        elif prefecture:
            pref_info = self.index['prefectures'].get(prefecture, {})
            files = []
            for city in pref_info.get('cities', []):
                city_path = f"{prefecture}/{city}"
                city_info = self.index['cities'].get(city_path, {})
                for dist in city_info.get('districts', []):
                    dist_path = f"{city_path}/{dist}"
                    files.extend(self.index['districts'].get(dist_path, {}).get('files', []))
            return files
        
        return []