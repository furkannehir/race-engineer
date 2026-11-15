#ifndef AppVersion
  #error AppVersion must be supplied with /DAppVersion=x.y.z
#endif

#define AppName "Pitward"
#define AppPublisher "Pitward contributors"
#define AppURL "https://github.com/furkannehir/race-engineer"
#define AppExeName "Pitward.exe"
#define AppUserModelId "io.github.furkannehir.pitward"

[Setup]
AppId={{D31DCB74-619A-4A92-911D-D3C2F32FE6D4}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=Pitward-Setup-{#AppVersion}-windows-x64
SetupIconFile=..\src\race_engineer\assets\pitward-brand-pack\icons\pitward.ico
UninstallDisplayIcon={app}\{#AppExeName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no

[Files]
Source: "..\build\package\dist\Pitward\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"; AppUserModelID: "{#AppUserModelId}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon; AppUserModelID: "{#AppUserModelId}"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\{#AppExeName}"; Parameters: "{code:LaunchParameters}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Explicit uninstall removes component/update caches; profiles and logs are retained.
; In-place upgrades do not run UninstallDelete.
Type: filesandordirs; Name: "{localappdata}\Pitward\components"
Type: filesandordirs; Name: "{localappdata}\Pitward\updates"
Type: filesandordirs; Name: "{app}"

[Code]
function LaunchParameters(Param: String): String;
var
  ConfigPath, SettingsPath: String;
begin
  Result := '';
  ConfigPath := ExpandConstant('{param:PITWARDCONFIG|}');
  SettingsPath := ExpandConstant('{param:PITWARDSETTINGS|}');
  if ConfigPath <> '' then
    Result := Result + ' --config "' + ConfigPath + '"';
  if SettingsPath <> '' then
    Result := Result + ' --settings "' + SettingsPath + '"';
end;
