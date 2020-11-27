# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

added_files = [
 	( 'PyShaderPlayground/ShaderPlayground.ui', 'PyShaderPlayground' ),
 	( 'PyShaderPlayground/VideoEncodingParams.ui', 'PyShaderPlayground' ),
 	( 'PyShaderPlayground/ResolutionDialog.ui', 'PyShaderPlayground' ),
 	( 'PyShaderPlayground/ProcessRunner.ui', 'PyShaderPlayground' )
]

a = Analysis(['PyShaderPlayground/__main__.py'],
             pathex=['D:/devel/sandbox/PyShaderPlayground'],
             binaries=[],
             datas=added_files,
             hiddenimports=['PySide2.QtXml'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='ShaderPlayground',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='ShaderPlayground')
