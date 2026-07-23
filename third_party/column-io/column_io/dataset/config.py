import hashlib
import os


STORAGE_URL_MAP = {}

_kIncColumnSplit = "\x1f"
class LakeConfig:
    def __init__(
        self,
        storageName,
        projectName,
        tableName,
        columnFamilyName,
        partitionSpec,
        startTime=None,
        endTime=None,
        useService=True,
        serviceName=None,
    ):
        self._storageName = storageName
        self._projectName = projectName
        self._tableName = tableName
        self._columnFamilyName = columnFamilyName
        self._partitionSpec = partitionSpec
        self._startTime = startTime
        self._endTime = endTime
        if startTime is None or startTime <= 0:
            self._startTime = -1
        if endTime is None or endTime <= 0:
            self._endTime = -1
        if serviceName is None:
            # if user not specify serviceName, serviceName will generated from lake table name
            hash_object = hashlib.sha256()
            hash_object.update((self._projectName + self._tableName).encode("utf-8"))
            digest = hash_object.hexdigest()[0:4]
            sub_name = self._tableName.replace("_", "-")
            sub_name = sub_name[0 : min(30, len(sub_name))]
            if sub_name.startswith("-"):
                sub_name = sub_name[1:]
            self._serviceName = "tcp:" + sub_name + "-" + digest + ".vipserver:80"
        else:
            self._serviceName = serviceName
        self._useService = useService
    
    @staticmethod
    def is_inc_path(path: str):
        """
        check if path contains inc table
        """
        if isinstance(path, str):
           return _kIncColumnSplit in path
        else:
           return _kIncColumnSplit.encode() in path

    def add_columnfamily(self, columnFamilyName):
      if self._columnFamilyName is None:
        self._columnFamilyName = columnFamilyName
      else :
        self._columnFamilyName = self._columnFamilyName+_kIncColumnSplit+columnFamilyName

        
    def get_v1_path(self):
        if self._storageName not in STORAGE_URL_MAP:
            raise ValueError("storage name is invaild: {}".format(self._storageName))
        name_list = []
        name_list.append(STORAGE_URL_MAP[self._storageName])
        if self._projectName is not None and self._projectName != "":
            name_list.append(self._projectName)
        if self._tableName is not None and self._tableName != "":
            name_list.append(self._tableName)
        if self._columnFamilyName is not None and self._columnFamilyName != "":
            name_list.append(self._columnFamilyName)
        if self._partitionSpec is not None and self._partitionSpec != "":
            name_list.append(self._partitionSpec)
        if self._useService == False:
            return "path=" + "/".join(name_list)
        else:
            return "path=" + "/".join(name_list) + ";serviceName=" + self._serviceName

    def get_v2_path(self, withTime=True):
        table_path = self.get_v1_path()
        if withTime == False:
            return "lake://" + table_path
        return (
            "lake://"
            + table_path
            + "|begin="
            + str(self._startTime)
            + ";end="
            + str(self._endTime)
        )
    
class LakeBatchConfig():
  def __init__(self, storageName, projectName, tableName, columnFamilyName, partitionSpec):
    self._storageName = storageName
    self._projectName = projectName
    self._tableName = tableName
    self._columnFamilyName = columnFamilyName
    self._partitionSpec = partitionSpec
    

  def get_v1_path(self):
    name_list = []
    if self._storageName not in STORAGE_URL_MAP:
      name_list.append(self._storageName)
    else:
      name_list.append(STORAGE_URL_MAP[self._storageName])
    if self._projectName is not None and self._projectName != "":
      name_list.append(self._projectName)
    if self._tableName is not None and self._tableName != "":
      name_list.append(self._tableName)
    if self._columnFamilyName is not None and self._columnFamilyName != "":
      name_list.append(self._columnFamilyName)
    if self._partitionSpec is not None and self._partitionSpec != "":
      name_list.append(self._partitionSpec)

    return "path=" + "/".join(name_list)
  
  def get_v2_path(self, withTime = True):
    table_path = self.get_v1_path()
    return "lake://" + table_path


def is_external_cluster():
    is_external = os.environ.get('IS_EXTERNAL_CLUSTER', None)
    return str(is_external).lower() == 'true'

